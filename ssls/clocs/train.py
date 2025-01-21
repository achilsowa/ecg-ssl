# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import torch.distributed as dist

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

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from src.utils.distributed import (
    init_distributed,
    AllReduceSum
)
from src.utils.logging import (
    CSVLogger,
    WANDBLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.datasets.ecgnet import make_ecgnet
from src.models import get_model

from src.utils.optimizers import (
    get_param_groups,
    init_opt,
)
from src.utils.helper import (
    init_local_config,
    clean_state_dict
)

from src.transforms import make_group_transforms as make_transforms
from ssls.simclr.objective import nt_xent_loss

# --
log_timings = True
log_freq = 25
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
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    projection_name = args['meta']['projection_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        torch.cuda.device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    use_grayscale = True if args['data']['in_chans'] == 1 else False
    gaussian_noise = args['data'].get('gaussian_noise', 0.5)
    sobel_derivative = args['data'].get('sobel_derivative', False)
    reverse=args['data'].get('reverse', False)
    invert=args['data'].get('invert', False)
    rand_wanderer=args['data'].get('rand_wanderer', False)
    baseline_wanderer=args['data'].get('baseline_wanderer', False)
    baseline_shift=args['data'].get('baseline_shift', False)
    em_noise=args['data'].get('em_noise', False)
    pl_noise=args['data'].get('pl_noise', False)
    time_out=args['data'].get('time_out', False)
    scale=args['data'].get('scale', False)
    
    # --
    batch_size = args['data']['batch_size']
    patch_size = args['data'].get('patch_size', None)  # patch-size for model training
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    clean = args['data']['clean']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    in_chans = args['data']['in_chans']
    x_label = args['data']['x_label']
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
        "gaussian_noise": gaussian_noise,
        "sobel_derivative": sobel_derivative,
        "reverse": reverse,
        "invert": invert,
        "rand_wanderer": rand_wanderer,
        "baseline_wanderer": baseline_wanderer,
        "baseline_shift": baseline_shift,
        "em_noise": em_noise,
        "pl_noise": pl_noise,
        "time_out": time_out,
        "scale": scale,
    }
    train_config = init_local_config(args['data']['train'], **train_data_kwargs)
    
    is_1d = x_label == "ecg"
    assert x_label == "ecg" or x_label == "img", f"x_label should be in [ecg|img] not {x_label}"
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    opt_strategy = args['optimization']['strategy']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    temperature = args['optimization'].get('temperature', 0.1)
    grad_clip = args['optimization'].get('grad_clip', 10.)

    # -- LOGGING
    folder = args['logging']['folder']
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    tag = args['logging']['write_tag']
    wandb_login =args['logging'].get('wandb_login', False)
    exp_project=args['logging']['project'] 
    exp_name=args['logging']['name'] 
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
                           ('%.5f', 'accuracy'),
                           ('%.5f', 'entropy'),
                           ('%d', 'time (ms)'))
    
    wandb_logger = WANDBLogger(
        login=wandb_login,
        dir=os.path.join(folder),
        project=exp_project, 
        name=exp_name, 
        id=exp_id)
    
    def log_csv_wandb(epoch, itr, loss, scores,  etime):
        acc = scores.get("accuracy")
        entropy = scores.get("entropy")
        logger.info('[%d, %5d]  (loss: %.3f) [acc: %.3f] [entropy: %.3f] [etime: %d]'
            % (epoch, itr, loss, acc, entropy, etime))
        csv_logger.log(epoch, itr, loss, acc, entropy, etime)
        wandb_logger.log({
            "epoch": epoch,
            "itr": itr,
            "loss": loss,
            "accuracy": acc,
            "entropy": entropy,
            "etime": etime
        })


    # -- init model
    encoder, projection = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        in_chans=in_chans,
        model_name=model_name,
        projection_name=projection_name
        )
    
    transform = make_transforms(x_label, train_config,)
    logger.info(f'transforms:\n{ transform[0] if isinstance(transform, list) else transform}')

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
    logger.info(f'Dataloader created... iterations per epoch: {ipe}')


    # -- init optimizer and scheduler
    params_groups = get_param_groups(encoder) + get_param_groups(projection)
    opt = init_opt(
        opt_strategy,
        param_groups=params_groups,
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
    if dist.is_available() and dist.is_initialized():
        encoder = DistributedDataParallel(encoder, static_graph=True)
        projection = DistributedDataParallel(projection, static_graph=True)

    
    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, projection, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            projection=projection,
            opt=opt.optimizer,
            scaler=opt.scaler)
        for _ in range(start_epoch*ipe):
            opt.lr_step()
            opt.wd_step()
        opt.scaler = scaler
        opt.optimizer = optimizer
    
    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'projection': projection.state_dict(),
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

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))
        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)
        loss, etime, scores = run_one_epoch(
            device=device,
            training=True,
            encoder=encoder,
            projection=projection,
            opt=opt,
            grad_clip=grad_clip,
            temperature=temperature,
            warmup=warmup,
            data_loader=unsupervised_loader,
            x_label=x_label,
            compute_metrics=True,
            rank=rank,
            epoch=epoch,
            use_bfloat16=use_bfloat16
        )
        log_csv_wandb(epoch + 1, 0, loss, scores, etime)
       
        save_checkpoint(epoch+1)


def run_one_epoch(
    device,
    training,
    encoder,
    projection,
    opt,
    grad_clip,
    data_loader,
    x_label,
    temperature,
    use_bfloat16,
    rank,
    epoch,
    warmup,
    compute_metrics=False
):
    compute_metrics = compute_metrics and rank == 0
    encoder.train(mode=training)
    projection.train(mode=training)
    loss_meter = AverageMeter()
    time_meter = AverageMeter()
    acc_meter = AverageMeter()
    entropy_meter = AverageMeter()
    for itr, data in enumerate(data_loader):
        
        def train_step():
            def load_inputs():
                # -- unsupervised data
                x = data[x_label].to(device, non_blocking=True) # from (B, C, ... last_dim)
                W = x.shape[-1]//2
                x_1, x_2 = x.split(W, -1)
                x = torch.cat([x_1, x_2], 0)  # to (2 * B, C, ..., last_dim/2)
                return x
            
            x = load_inputs()
            def forward():
                if training:
                    outputs = encoder(x)
                    outputs = projection(outputs)
                else:
                    with torch.no_grad():
                        outputs = encoder(x)
                        outputs = projection(outputs)
                return outputs            

            if torch.cuda.is_available():        
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
                    Z = forward()
            else:
                Z = forward()
        
            loss, logits, labels = nt_xent_loss(Z, temp=temperature, rank=rank)
            loss = AllReduceSum.apply(loss)
            loss_meter.update(loss.item())
            if training:
                if use_bfloat16:
                    opt.scaler.scale(loss).backward()
                    opt.scaler.unscale_(opt.optimizer)
                    if (epoch > warmup) and (grad_clip is not None):
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
                    opt.scaler.step(opt.optimizer)
                    opt.scaler.update()
                else:
                    loss.backward()
                    if (epoch > warmup) and (grad_clip is not None):
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
                    opt.optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                opt.optimizer.zero_grad()

            if compute_metrics:
                scores = calculate_metrics(logits, labels)
            else:
                scores = {}
            
            new_lr = new_wd = None
            if training:
                new_lr = opt.lr_step()
                new_wd = opt.wd_step()

            return (float(loss), scores, new_lr, new_wd, grad_stats)
            
            
        (loss, scores, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
        loss_meter.update(loss)
        time_meter.update(etime)
        acc_meter.update(scores.get("accuracy", 0))
        entropy_meter.update(scores.get("entropy", 0))
        
        if itr % log_freq == 0:
            logger.info('[%d, %5d]  (loss: %.3f) [mem: %.2e] [wd: %.2e] [lr: %.2e] '
                % (epoch + 1, itr, loss, torch.cuda.max_memory_allocated() / 1024.**2, _new_wd, _new_lr))
            if grad_stats is not None:
                logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                    % (epoch + 1, itr, grad_stats.first_layer, grad_stats.last_layer, grad_stats.min, grad_stats.max))

    
    return loss_meter.avg, time_meter.avg, {"accuracy": acc_meter.avg, "entropy": entropy_meter.avg}
    
    #IN CASE A PORT IS ALREADY IN USE, RANDOMLY SAMPLE AN INTEGER IN [0, 5] , ADD IT TO THE CURRENT PORT AND TRY AGAIN


def load_checkpoint(
    device,
    r_path,
    encoder,
    projection,
    opt,
    scaler
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading encoder
        encoder = load_pretrained(encoder=encoder, pretrained=r_path, checkpoint_key='encoder')
    
        # -- loading classifier
        projection_dict = clean_state_dict(projection, checkpoint['projection'])
        msg = projection.load_state_dict(projection_dict)
        logger.info(f'loaded pretrained projection from epoch {epoch} with msg: {msg}')

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

    return encoder, projection, opt, scaler, epoch


def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='encoder'
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
    model_name,
    patch_size,
    crop_size,
    in_chans,
    projection_name,
    num_classes=128,
):
    encoder = get_model(model_name,
        input_size=[crop_size],
        in_chans=in_chans,
        patch_size=patch_size,)
    
    encoder.to(device)
    projection = get_model(projection_name , num_classes=num_classes).to(device)
    return encoder, projection


def calculate_metrics(logits, labels):
    """Calculate accuracy and entropy given logits and labels."""
    with torch.no_grad():
        _, preds = torch.max(logits, dim=1)
        _, labels = torch.max(labels, dim=1)
        acc = (preds == labels).float().mean()
    
        prob = F.softmax(logits, dim=1)
        entropy = -torch.mean(torch.sum(prob * torch.log(prob + 1e-8), dim=1))

    return {"accuracy": float(acc), "entropy": float(entropy)}

