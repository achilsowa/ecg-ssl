# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import pprint

import numpy as np

import torch
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel

from src.models import get_model
from src.datasets.ecgnet import make_ecgnet
from src.transforms import make_group_transforms as make_transforms
from src.utils.helper import (
    init_local_config,
    clean_state_dict
)
from src.utils.optimizers import (
    EarlyStopping,
    init_opt,
    get_param_groups,
)
from src.utils.distributed import (
    init_distributed,
    AllReduce,
    AllGather
)
from src.utils.metrics import (
    calculate_metrics, 
    metrics_headers
)
from src.utils.logging import (
    AverageMeter,
    CSVLogger
)

# --
log_timings = True
log_freq = 20
checkpoint_freq = 5
# --

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- PRETRAIN
    args_pretrain = args_eval.get('model')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name', None)
    adapter_name = args_pretrain.get('adapter_name', None)
    classifier_name = args_pretrain.get('classifier_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    pretrained_path = None if pretrain_folder is None or ckp_fname is None else os.path.join(pretrain_folder, ckp_fname)
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- DATA
    args_data = args_eval.get('data')
    num_workers = args_data.get('num_workers', 10)
    use_grayscale = True if args_data['in_chans'] == 1 else False
    num_classes = args_data.get('num_classes')
    resolution = args_data.get('resolution', 2500)
    clean = args_data.get('clean')
    in_chans = args_data.get('in_chans')
    x_label = args_data.get('x_label')
    y_labels = args_data.get('y_labels', [])
    categories = args_data.get('categories', {})
    data_kwargs = {"x_label": x_label, "clean": clean, "resolution": resolution, "y_labels": y_labels}
    eval_data_kwargs = {
        "x_label": x_label, 
        "y_labels": y_labels,
        "clean": clean, 
        "random_crop": False,
        "center_crop": True,
        "crop_size": resolution, 
        "resolution": resolution,
        "gray_scale": use_grayscale,
    }
    should_train, should_val, should_test = [key in args_data for key in ['train', 'val', 'test']]
    assert x_label == "ecg" or x_label == "img", f"x_label should be in [ecg|img] not {x_label}"

    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    opt_strategy = args_opt.get('strategy', None) # in case we only test, no need for optimization
    batch_size = args_opt.get('batch_size')
    num_epochs = args_opt.get('num_epochs', None)
    wd = args_opt.get('weight_decay', None)
    start_lr = args_opt.get('start_lr', None)
    lr = args_opt.get('lr', None)
    final_lr = args_opt.get('final_lr', None)
    warmup = args_opt.get('warmup', None)
    patience = args_opt.get('patience', 10)
    mode = args_opt.get('mode', 'min')
    factor = args_opt.get('factor', 0.1)
    use_bfloat16 = args_opt.get('use_bfloat16', False)
    early_stop = args_opt.get('early_stop', False)
    early_stop_patience = args_opt.get('early_stop_patience', 5)
    early_stop_mindelta = args_opt.get('min_delta', 0.001)

    # -- EXPERIMENT-ID/TAG (optional)
    resume_checkpoint = args_eval.get('resume_checkpoint', False) or resume_preempt
    read_checkpoint = args_eval.get('read_checkpoint', None)
    test_only = args_eval.get('test_only', False)
    eval_tag = args_eval.get('tag', None)
    

    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        #device = torch.device('cuda:0')
        #torch.cuda.set_device(device)
        device = torch.device('cuda')
        torch.cuda.device(device)
        

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, 'ecg_classification/')
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    log_test_file = os.path.join(folder, f'{tag}_test_r{rank}.csv')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    r_path = latest_path if read_checkpoint is None else read_checkpoint

    # -- make csv_logger
    m_headers = metrics_headers(y_labels, categories)
    headers = [('%d', 'epoch'), ('%.4f', 'train_loss'), ('%.4f', 'val_loss')]
    for h in m_headers:
        headers += [('%.3f', f'{h}_tm'), ('%.3f', f'{h}_sk')]
    csv_logger = CSVLogger(log_file, *headers)
    csv_testlogger = CSVLogger(log_test_file, *headers)
    
    #wandb_logger = WANDBLogger(login=wandb_login,dir=os.path.join(folder),project=exp_project, name=exp_name, id=exp_id)
    def log_csv_wandb(epoch, train_loss=0, val_loss=0, scores={"tm": {}, "sk": {}}, is_test=False):
        logger = csv_testlogger if is_test else csv_logger
        prefix = 'test' if is_test else 'val'

        data = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,}
        for h in m_headers:
            data[f'{h}_tm'] = scores["tm"][h] if h in scores["tm"] else 0
            data[f'{h}_sk'] = scores["sk"][h] if h in scores["sk"] else 0
        
        m_args = [data[h[1]] for h in headers]    
        logger.log(*m_args)

        k_args = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,}
        for h in headers[3:]:
            k_args[f'{prefix}_{h[1]}'] = data[h[1]]
        
        #wandb_logger.log(k_args)


    # Initialize model

    # -- pretrained encoder (frozen)
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        in_chans=in_chans,
        checkpoint_key=checkpoint_key)
    
    # -- init classifier
    classifier = init_classifier(device, classifier_name, num_classes=num_classes)
    classifier = init_adapter(device, adapter_name, encoder=encoder, classifier=classifier)

    
    # -- init data-loaders/samplers
    if should_train:
        train_config = init_local_config(args_data['train'], **eval_data_kwargs)
        val_config = init_local_config(args_data['val'], **eval_data_kwargs)
        train_transform = make_transforms(x_label, train_config)    
        val_transform = make_transforms(x_label, val_config)
        _, train_loader, _ = make_ecgnet(
            transform=train_transform,
            batch_size=batch_size,
            config=train_config,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,)
        _, val_loader, _ = make_ecgnet(
            transform=val_transform,
            batch_size=batch_size,
            config=val_config,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,)
        ipe = len(train_loader)
        logger.info(f'Dataloader created... iterations per epoch: {ipe}')
    if should_test:
        test_config = init_local_config(args_data['test'], **eval_data_kwargs)
        test_transform = make_transforms(x_label, test_config)
        _, test_loader, _ = make_ecgnet(
            transform=test_transform,
            batch_size=batch_size,
            config=test_config,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,)

    # -- optimizer and scheduler
    if should_train:
        param_groups = get_param_groups(encoder) + get_param_groups(classifier)
        opt = init_opt(
            opt_strategy,
            param_groups,
            wd=wd,
            start_lr=start_lr,
            ref_lr=lr,
            final_lr=final_lr,
            iterations_per_epoch=ipe,
            warmup=warmup,
            num_epochs=num_epochs,
            patience=patience,
            mode=mode,
            factor=factor,
            use_bfloat16=use_bfloat16,
            ipe_scale=1.)
        
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        encoder = DistributedDataParallel(encoder, static_graph=True)
        classifier = DistributedDataParallel(classifier, static_graph=True)

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint:
        encoder, classifier, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=r_path,
            encoder=encoder,
            classifier=classifier,
            opt=opt.optimizer if should_train else None,
            scaler=opt.scaler if should_train else None)
        if should_train:
            for _ in range(start_epoch*ipe):
                opt.lr_step()
                opt.wd_step()
            opt.scaler = scaler
            opt.optimizer = optimizer

    def save_checkpoint(epoch, best_checkpoint=False):
        save_dict = {
            'encoder': encoder.state_dict(),
            'classifier': classifier.state_dict(),
            'opt': opt.optimizer.state_dict(),
            'scaler': None if opt.scaler is None else opt.scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))
            if best_checkpoint:
                torch.save(save_dict, save_path.format(epoch='best'))

    # TRAIN LOOP
    if should_train:
        early_stopper = EarlyStopping(patience=early_stop_patience, min_delta=early_stop_mindelta)
        best_val_loss = np.inf
        for epoch in range(start_epoch, num_epochs):
            logger.info('Epoch %d' % (epoch + 1))
            train_loss, _ = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                classifier=classifier,
                opt=opt,
                data_loader=train_loader,
                x_label=x_label,
                y_labels=[],
                categories={},
                compute_metrics=False,
                rank=rank,
                epoch=epoch,
                use_bfloat16=use_bfloat16)

            val_loss, val_scores = run_one_epoch(
                device=device,
                training=False,
                encoder=encoder,
                classifier=classifier,
                opt=opt,
                data_loader=val_loader,
                x_label=x_label,
                y_labels=y_labels,
                categories=categories,
                compute_metrics=True,
                rank=rank,
                epoch=epoch,
                use_bfloat16=use_bfloat16)
            
            opt.lr_step_epoch(val_loss)
            opt.wd_step_epoch()

            logger.info('[%5d] len_all_items: %d train: %.3f val: %.3f' 
                        % (epoch + 1, val_scores["len_all_items"], train_loss, val_loss))
            if rank == 0:
                log_csv_wandb(epoch + 1,  train_loss, val_loss, val_scores)
            best_val_loss = min(best_val_loss, float(val_loss))
            
            save_checkpoint(epoch + 1, best_checkpoint=(best_val_loss == float(val_loss)))
            early_stopper(val_loss)
            if early_stop and early_stopper.early_stop:
                logger.info('Early stopping triggered at epoch %d' % (epoch + 1))
                break

    # TEST LOOP
    if should_test:
        logger.info('TESTING')
        test_epoch = 0
        encoder.eval()
        classifier.eval()
        test_loss, test_scores = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifier=classifier,
            opt=None,
            data_loader=test_loader,
            x_label=x_label,
            y_labels=y_labels,
            categories=categories,
            compute_metrics=True,
            rank=rank,
            epoch=test_epoch,
            use_bfloat16=use_bfloat16)

        logger.info('[%5d] len_all_items: %d test: %.3f' 
                    % (test_epoch, test_scores["len_all_items"], test_loss))
        if rank == 0:
            log_csv_wandb(test_epoch, 0., test_loss, test_scores, is_test=True)



def run_one_epoch(
    device,
    training,
    encoder,
    classifier,
    opt,
    data_loader,
    x_label,
    y_labels,
    categories,
    use_bfloat16,
    rank,
    epoch,
    compute_metrics=False
):
    compute_metrics = compute_metrics and rank == 0
    classifier.train(mode=training)
    encoder.train(mode=training)
    criterion = torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss()
    top1_meter = AverageMeter()
    loss_meter = AverageMeter()
    all_outputs = []
    all_labels = []
    for itr, data in enumerate(data_loader):

        if training:
            opt.lr_step()
            opt.wd_step()

        def forward(x):
            if training:
                outputs = encoder(x)
                outputs = classifier(outputs)
            else:
                with torch.no_grad():
                    outputs = encoder(x)
                    outputs = classifier(outputs)
            return outputs            

        if torch.cuda.is_available():        
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
                x, labels = data[x_label].to(device), data["labels"].to(device)
                outputs = forward(x)
        else:
            x, labels = data[x_label].to(device), data["labels"].to(device)
            outputs = forward(x)
        
        
        loss = criterion(outputs, labels.to(dtype=outputs.dtype))
        loss = AllReduce.apply(loss)
        
        gathered_outputs = AllGather.apply(outputs)
        gathered_labels = AllGather.apply(labels)
        if compute_metrics:
            all_outputs.extend(gathered_outputs.float().detach().cpu().numpy())
            all_labels.extend(gathered_labels.float().detach().cpu().numpy())

        loss_meter.update(loss.item())

        if training:
            if use_bfloat16:
                opt.scaler.scale(loss).backward()
                opt.scaler.unscale_(opt.optimizer)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                opt.scaler.step(opt.optimizer)
                opt.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                opt.optimizer.step()
            opt.optimizer.zero_grad()

        
        
        if itr % log_freq == 0:
            logger.info('[%d, %5d]  (loss: %.3f) [mem: %.2e]'
                        % (epoch + 1, itr,  loss, torch.cuda.max_memory_allocated() / 1024.**2))


    if compute_metrics:    
        all_labels = np.array(all_labels)
        all_outputs = np.array(all_outputs)
        scores = calculate_metrics(all_labels, all_outputs, x.device, y_labels, categories)
    else:
        scores = {}
    
    scores["len_all_items"] = len(all_labels)
    return loss_meter.avg, scores
    
    #IN CASE A PORT IS ALREADY IN USE, RANDOMLY SAMPLE AN INTEGER IN [0, 5] , ADD IT TO THE CURRENT PORT AND TRY AGAIN

def load_checkpoint(
    device,
    r_path,
    encoder,
    classifier,
    opt,
    scaler
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading encoder
        encoder = load_pretrained(encoder=encoder, pretrained=r_path, checkpoint_key='encoder')
    
        # -- loading classifier
        classifier_dict = clean_state_dict(classifier, checkpoint['classifier'])
        msg = classifier.load_state_dict(classifier_dict)
        logger.info(f'loaded pretrained classifier from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        if opt is not None:
            opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, classifier, opt, scaler, epoch


def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='target_encoder'
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = clean_state_dict(encoder, pretrained_dict)
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder

def init_model(
    device,
    pretrained,
    model_name,
    patch_size,
    crop_size,
    in_chans,
    checkpoint_key='target_encoder'
):
    encoder = get_model(model_name,
        input_size=[crop_size],
        in_chans=in_chans,
        patch_size=patch_size,)
    
    encoder.to(device)
    if pretrained is not None:
        encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    
    return encoder

def init_classifier(
    device,
    classifier_name,
    num_classes,
):
    # -- init classifier
    classifier = get_model(classifier_name, num_classes=num_classes).to(device)
    return classifier

def init_adapter(
    device,
    adapter_name,
    encoder,
    classifier,
):
    # -- init adapter, and update classifier if needed
    if adapter_name is not None:
        adapter = get_model(adapter_name, encoder=encoder, classifier=classifier)
        children = list(adapter.children())
        classifier = torch.nn.Sequential(*children, classifier).to(device)
    return classifier
