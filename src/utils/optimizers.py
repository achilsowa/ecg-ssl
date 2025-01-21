import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from abc import ABC

from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
    LinearWarmupCosineAnnealingLR,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class Optimizer(ABC):
    scheduler: None
    scaler: None

    def __init__(
            self, 
            param_groups, 
            iterations_per_epoch,
            num_epochs,
            wd=1e-6,
            final_wd=1e-6,
            use_bfloat16=False,
            ipe_scale=1.25,
            **kwargs) -> None:
        self.optimizer = optim.AdamW(param_groups)
        self.wd_scheduler = CosineWDSchedule(
            self.optimizer,
            ref_wd=wd,
            final_wd=final_wd,
            T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
        self.scaler = torch.amp.GradScaler('cuda') if use_bfloat16 and torch.cuda.is_available() else None


    def __call__(self):
        return self.optimizer, self.scaler, self.scheduler, self.wd_scheduler

    def step(self, *args, **kwargs):
        return self.optimizer.step()
    
    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad()
    
    def lr_step(self, *args, **kwargs):
        return self.scheduler.step()
    
    def wd_step(self, *args, **kwargs):
        if self.wd_scheduler is not None:
            return self.wd_scheduler.step()
        return None

    def lr_step_epoch(self, *args, **kwargs):
        pass
    
    def wd_step_epoch(self, *args, **kwargs):
        pass

class WarmupCosineOptimizer(Optimizer):
    def __init__(
            self,
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
            ipe_scale=1.25,
            **kwargs) -> None:
        super().__init__(
            param_groups, 
            iterations_per_epoch=iterations_per_epoch,
            num_epochs=num_epochs,
            wd=wd, 
            final_wd=final_wd, 
            use_bfloat16=use_bfloat16, 
            ipe_scale=ipe_scale, 
            **kwargs)
        self.scheduler = WarmupCosineSchedule(
            self.optimizer,
            warmup_steps=int(warmup*iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    
class ReduceLROnPlateauOptimizer(Optimizer):
    def __init__(
            self, 
            param_groups, 
            iterations_per_epoch,
            num_epochs,
            wd=1e-6,
            final_wd=1e-6,
            mode="min",
            factor=0.1,
            patience=5,
            final_lr=0.0,
            use_bfloat16=False,
            ipe_scale=1.25,
            **kwargs
            ) -> None:
        super().__init__(
            param_groups, 
            iterations_per_epoch=iterations_per_epoch,
            num_epochs=num_epochs,
            wd=wd, 
            final_wd=final_wd, 
            use_bfloat16=use_bfloat16, 
            ipe_scale=ipe_scale, 
            **kwargs)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=final_lr)
    
    def lr_step(self, *args, **kwargs):
        pass
    
    def lr_step_epoch(self, monitor_val, *args, **kwargs):
        return self.scheduler.step(monitor_val)
    
class LinearWarmupCosineAnnealingOptimizer(Optimizer):
    def __init__(
            self, 
            param_groups, 
            iterations_per_epoch, 
            start_lr,
            warmup,
            num_epochs, 
            wd=1e-6, 
            final_wd=1e-6, 
            use_bfloat16=False, 
            ipe_scale=1.25, 
            **kwargs) -> None:
        super().__init__(
            param_groups, 
            iterations_per_epoch=iterations_per_epoch, 
            num_epochs=num_epochs, 
            wd=wd, 
            final_wd=final_wd, 
            use_bfloat16=use_bfloat16, 
            ipe_scale=ipe_scale, 
            **kwargs)
        
        max_epochs = num_epochs * iterations_per_epoch
        warmup_epochs = warmup * iterations_per_epoch
        self.scheduler = LinearWarmupCosineAnnealingLR(
            self.optimizer, 
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=start_lr
        )
    
    def lr_step(self, *args, **kwargs):
        self.scheduler.step()
        return sum(self.scheduler._last_lr)/len(self.scheduler._last_lr)



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# REMOVE EMPTY GROUP [ie all p.requiregs_grad is true so the final array is empty]


def get_param_groups(model: nn.Module):
    param_groups = [
        {
            'params': (p for n, p in model.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1) and p.requires_grad),
        }, {
            'params': (p for n, p in model.named_parameters()
                       if (('bias' in n) or (len(p.shape) == 1) or ('bn' in n)) and p.requires_grad),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]
    return param_groups



def init_opt(
    strategy,
    param_groups,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mode="min",
    factor=0.1,
    patience=10,        
    use_bfloat16=False,
    ipe_scale=1.25,
) -> Optimizer:
    logger.info('Using AdamW')
    if strategy == "warmup-cosine":
        optimizer = WarmupCosineOptimizer(
            param_groups=param_groups,
            iterations_per_epoch=iterations_per_epoch,
            start_lr=start_lr,
            ref_lr=ref_lr,
            warmup=warmup,
            num_epochs=num_epochs,
            wd=wd,
            final_wd=final_wd,
            final_lr=final_lr,
            use_bfloat16=use_bfloat16,
            ipe_scale=ipe_scale
        )
    elif strategy == "reduce-on-plateau":
        optimizer = ReduceLROnPlateauOptimizer(
            param_groups=param_groups,
            iterations_per_epoch=iterations_per_epoch,
            num_epochs=num_epochs,
            wd=wd,
            final_wd=final_wd,
            mode=mode,
            factor=factor,
            patience=patience,
            final_lr=final_lr,
            use_bfloat16=use_bfloat16,
            ipe_scale=ipe_scale
        )
    elif strategy == "linear-warmup-cosine":
        optimizer = LinearWarmupCosineAnnealingOptimizer(
            param_groups=param_groups,
            iterations_per_epoch=iterations_per_epoch,
            start_lr=start_lr,
            warmup=warmup,
            num_epochs=num_epochs,
            wd=wd,
            final_wd=final_wd,
            use_bfloat16=use_bfloat16,
            ipe_scale=ipe_scale
        )
    else:
        raise RuntimeError(f"strategy should be one of [warmup-cosine|reduce-on-plateau|linear-warmup-cosine] not {strategy}")
    
    return optimizer
    
   