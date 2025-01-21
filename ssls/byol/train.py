# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import torch.distributed

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.models import get_model
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_

from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    WANDBLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.datasets.ecgnet import make_ecgnet

from src.utils.helper import (
    init_local_config,
    clean_state_dict)
from src.utils.optimizers import get_param_groups
from src.transforms import make_group_transforms as make_transforms

# --
log_timings = True
log_freq = 10
checkpoint_freq = 25
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    args_meta = args['meta']
    use_bfloat16 = args_meta.get('use_bfloat16')
    model_name = args_meta.get('model_name')
    projection_name = args_meta.get('projection_name')
    predictor_name = args_meta.get('predictor_name')
    load_model = args_meta.get('load_checkpoint') or resume_preempt
    r_file = args_meta.get('read_checkpoint')
    #copy_data = args_meta.get('copy_data']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        torch.cuda.device(device)

    # -- DATA
    args_data = args['data']
    use_gaussian_blur = args_data.get('use_gaussian_blur')
    use_horizontal_flip = args_data.get('use_horizontal_flip')
    use_color_distortion = args_data.get('use_color_distortion')
    color_jitter = args_data.get('color_jitter_strength')
    use_grayscale = True if args_data.get('in_chans') == 1 else False
    # --
    batch_size = args_data.get('batch_size')
    patch_size = args_data.get('patch_size', None)
    pin_mem = args_data.get('pin_mem')
    num_workers = args_data.get('num_workers')
    #root_path = args_data.get('root_path']
    #image_folder = args_data.get('image_folder']
    clean = args_data.get('clean')
    crop_size = args_data.get('crop_size')
    crop_scale = args_data.get('crop_scale')
    in_chans = args_data.get('in_chans')
    x_label = args_data.get('x_label')
    train_data_kwargs = {
        "x_label": x_label, 
        "clean": clean, 
        "crop_size": crop_size, 
        "crop_scale": crop_scale, 
        "gaussian_blur": use_gaussian_blur,
        "horizontal_flip": use_horizontal_flip,
        "color_distortion":use_color_distortion,
        "gray_scale": use_grayscale,
        "color_jitter": color_jitter,
    }
    train_config = init_local_config(args_data.get('train'), **train_data_kwargs)
    assert x_label == "ecg" or x_label == "img", f"x_label should be in [ecg|img] not {x_label}"
    # --

    
    # -- OPTIMIZATION
    args_opt = args['optimization']
    ema = args_opt.get('ema')
    ipe_scale = args_opt.get('ipe_scale')  # scheduler scale factor (def: 1.0)
    wd = float(args_opt.get('weight_decay'))
    final_wd = float(args_opt.get('final_weight_decay'))
    num_epochs = args_opt.get('epochs')
    warmup = args_opt.get('warmup')
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')

    # -- LOGGING
    args_log = args['logging']
    folder = args_log.get('folder')
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    tag = args_log.get('write_tag')
    wandb_login =args['logging'].get('wandb_login', False)
    exp_project=args_log.get('project')
    exp_name=args_log.get('name')
    exp_id = args['logging'].get('run_id', None)


    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%d', 'time (ms)'))
    
    wandb_logger = WANDBLogger(
        login=wandb_login,
        dir=os.path.join(folder),
        project=exp_project, 
        name=exp_name, 
        id=exp_id)
    
    def log_csv_wandb(epoch, itr, loss, etime):
        csv_logger.log(epoch, itr, loss, etime)
        wandb_logger.log({
            "epoch": epoch,
            "itr": itr,
            "loss": loss,
            "etime": etime
        })


    # -- init model
    encoder, projection, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        in_chans=in_chans,
        model_name=model_name,
        projection_name=projection_name,
        predictor_name=predictor_name
    )
    target_encoder = copy.deepcopy(encoder)
    target_projection = copy.deepcopy(projection)


    transform = make_transforms(x_label, train_config,)

    # -- init data-loaders/samplers
    # _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
    _, unsupervised_loader, unsupervised_sampler = make_ecgnet(
        transform=transform,
        batch_size=batch_size,
        pin_mem=pin_mem,
        config=train_config,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
    )
    ipe = len(unsupervised_loader)

    # -- init optimizer and scheduler
    param_groups = get_param_groups(encoder) + get_param_groups(projection) + get_param_groups(predictor)
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        param_groups=param_groups,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        encoder = DistributedDataParallel(encoder, static_graph=True)
        projection = DistributedDataParallel(projection, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)
        target_projection = DistributedDataParallel(target_projection)
    for p in target_encoder.parameters():
        p.requires_grad = False
    for p in target_projection.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, projection, predictor, target_encoder, target_projection, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            projection=projection,
            predictor=predictor,
            target_encoder=target_encoder,
            target_projection=target_projection,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'projection': projection.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'target_projection': target_projection.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))
        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, data in enumerate(unsupervised_loader):
            def load_inputs():
                # -- unsupervised data
                x = data[x_label].to(device, non_blocking=True) # from (B, C, ... 2*last_dim)
                W = x.shape[-1]//2
                x_1, x_2 = x.split(W, -1)
                x = torch.cat([x_1, x_2], 0)  # to (2 * B, C, ...)
                return x
           
            x = load_inputs()
            
            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(x)
                        z = target_projection(h)
                        return F.normalize(z, dim=1)

                def forward_context():
                    h = encoder(x)
                    z = projection(h)
                    q = predictor(z)
                    return F.normalize(q, dim=1)

                def loss_fn(z, q):
                    # I'll replace by paper BYOL loss in a second test loop
                    # loss = F.smooth_l1_loss(z, q) 
                    loss = torch.linalg.vector_norm(z-q, dim=1).mean()
                    return loss
                
                def forward():
                    h = forward_target()
                    z = forward_context()
                    B = z.shape[0]//2
                    h1, h2 = h.split(B, 0)
                    z1, z2 = z.split(B, 0)
                    loss = loss_fn(z1, h1) + loss_fn(z2, h2)
                    loss = AllReduce.apply(loss)
                    return loss                    

                # Step 1. Forward
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                        loss = forward()
                else:
                    loss = forward()

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder and target projection
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
                    for param_q, param_k in zip(projection.parameters(), target_projection.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                log_csv_wandb(epoch + 1, itr, loss, etime)

                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)



def load_checkpoint(
    device,
    r_path,
    encoder,
    projection,
    predictor,
    target_encoder,
    target_projection,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']
        def load_state_dict(model, key):
            if model is not None:
                pretrained_dict = clean_state_dict(model, checkpoint[key])
                msg = model.load_state_dict(pretrained_dict)
                logger.info(f'loaded pretrained {key} from epoch {epoch} with msg: {msg}')
            

        # -- loading models
        load_state_dict(encoder, 'encoder')
        load_state_dict(projection, 'projection')
        load_state_dict(predictor, 'predictor')
        load_state_dict(target_encoder, 'target_encoder')
        load_state_dict(target_projection, 'target_projection')
        
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

    return encoder, projection, predictor, target_encoder, target_projection, opt, scaler, epoch


def init_model(
    device,
    model_name,
    projection_name,
    predictor_name,
    patch_size=None,
    crop_size=224,
    in_chans=3,
):
    encoder = get_model(model_name,
        input_size=[crop_size],
        in_chans=in_chans,
        patch_size=patch_size,)
    projection = get_model(projection_name)
    predictor = get_model(predictor_name)

    encoder.to(device)
    projection.to(device)
    predictor.to(device)
    logger.info(encoder)

    return encoder, projection, predictor


def init_opt(
    param_groups,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler




class ContrastiveTransform:
    def __init__(self, transform) -> None:
        self.transform = transform
    def __call__(self, x):
        return torch.cat([self.transform(x) for _ in range(2)], -1)
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n"
        for t in [self.transform]:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string