# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import math


def gpu_timer(closure, log_timings=True):
    """ Helper to time gpu-time to execute closure() """
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


class CSVLogger(object):
    def __init__(self, fname, *argv):
        self.fname = fname
        self.argv = argv
        self.types = []
        self.file_created = False
        
    def create_file(self):
        argv = self.argv
        # -- print headers
        with open(self.fname, '+a') as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(f'"{v[1]}"', end=',', file=f)
                else:
                    print(f'"{v[1]}"', end='\n', file=f)
        self.file_created = True

    def log(self, *argv):
        if not self.file_created:
            self.create_file()

        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)

class WANDBLogger(object):

    def __init__(self, login, dir=None, project=None, name=None, id=None, config={}):
        # -- check wandb support
        self.logger = False
        if login:
            import wandb
            wandb.login()
            self.logger = wandb.init(dir=dir, project=project, name=name, config=config, id=id)

        
    def log(self, data):
        if self.logger:
            self.logger.log(data)

            
class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        if not math.isnan(val):
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats


def adamw_logger(optimizer):
    """ logging magnitude of first and second momentum buffers in adamw """
    # TODO: assert that optimizer is instance of torch.optim.AdamW
    state = optimizer.state_dict().get('state')
    exp_avg_stats = AverageMeter()
    exp_avg_sq_stats = AverageMeter()
    for key in state:
        s = state.get(key)
        exp_avg_stats.update(float(s.get('exp_avg').abs().mean()))
        exp_avg_sq_stats.update(float(s.get('exp_avg_sq').abs().mean()))
    return {'exp_avg': exp_avg_stats, 'exp_avg_sq': exp_avg_sq_stats}