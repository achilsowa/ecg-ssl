from logging import getLogger

import torch
import  torch.nn.functional as F
import random
import math

_GLOBAL_SEED = 0
logger = getLogger()



class RandomApply:
    """Apply randomly a transformation with a given probability.
    Args:
        transform 
        p (float): probability
    """

    def __init__(self, transform, p=0.5):
        super().__init__()
        self.transform = transform
        self.p = p

    def __call__(self, x):
        if self.p < torch.rand(1):
            return x
        return self.transform(x)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    p={self.p}"
        for t in [self.transform]:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class GaussianBlur:
    def __init__(self, kernel_sizes=(3, 5, 7), sigma=1):
        self.kernel_sizes = kernel_sizes
        self.sigma = sigma

    def __call__(self, x):
        kernel_size = random.choice(self.kernel_sizes)
        kernel = torch.arange(kernel_size, device=x.device) - kernel_size // 2
        kernel = torch.exp(-0.5 * (kernel.float() / self.sigma) ** 2)
        # Normalize the kernel
        kernel /= kernel.sum()
        return  apply_filter(x, kernel)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma}, ksizes={self.kernel_sizes})"

class SobelDerivative:
    def __call__(self, x:  torch.Tensor) -> torch.Tensor:
        """ Apply a filter blur with kernel `kernel` to the sample

        @param sample (torch.FloatTensor of shape (b, c, w)): ecg samples
        @return blurred sample
        """
        kernel = torch.tensor([-1, 0., 1])
        return apply_filter(x, kernel)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


class RandomCrop:
    def __init__(self,  size):
        super().__init__()
        self.size = size

    def __call__(self, x):
        x = x.clone()
        length = x.shape[-1]
        size = self.size
        start_idx = torch.randint(0, length - size + 1, size=(1,)).item()
        x = x[..., start_idx : start_idx + size]
        return x
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

class RandomResizedCrop:
    def __init__(self,  size, scale=(0.3, 1.0), interpolation="linear"):
        super().__init__()
        self.size = size
        self.scale = scale
        self.interpolation = interpolation
    def __call__(self, x):
        length = x.shape[-1]
        scale = self.scale
        size = self.size
        mode = self.interpolation
        x = x.unsqueeze(0)
        target_length = int(round(length * torch.empty(1).uniform_(scale[0], scale[1]).item()))
        start_idx = torch.randint(0, length - target_length + 1, size=(1,)).item()
        x = F.interpolate(x[..., start_idx : start_idx + target_length], size=size, mode=mode)
        return x.squeeze(0)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, scale={self.scale}, mode={self.interpolation})"

class Normalize:
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, x):
        """
        Args:
            x (Tensor): Tensor signal to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if not x.is_floating_point():
            raise TypeError(f"Input tensor should be a float tensor. Got {x.dtype}.")

        if x.ndim < 2:
            raise ValueError(
                f"Expected tensor to be a tensor signal of size (..., C, L). Got tensor.size() = {x.size()}"
            )

        if not self.inplace:
            x = x.clone()

        dtype = x.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=x.device)
        if (std == 0).any():
            raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
        if mean.ndim == 1:
            mean = mean.view(-1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1)
        return x.sub_(mean).div_(std)


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class Scale(object):
    """Deprecated. prefer Channel Resize"""
    def __init__(self, max_factor=5):
        super().__init__()
        self.max_factor = max_factor

    def __call__(self, x):
        factor = random.randint(1, self.max_factor) / 10
        return torch.mul(factor, x)
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_factor={self.max_factor})"

 
class TimeWarp(object): 
    """Currently supports only stretching"""

    def __init__(self, ratio=(0.1, 0.9)):
        super().__init__()
        self.ratio = ratio

    def __call__(self, x):
        ratio = random.uniform(*self.ratio)
        x = x.unsqueeze(0)
        x = F.interpolate(F.interpolate(x, scale_factor=ratio, mode="linear"), size=x.shape[-1], mode="linear")
        return x.squeeze(0)
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ratio={self.ratio})"

class TimeOut:
    """ replace random crop by zeros
    """
    def __init__(self, crop_ratio=(0.0, 0.5)):
        super(TimeOut, self).__init__()
        self.crop_ratio = crop_ratio

    def __call__(self, x):
        x = x.clone()
        C, W = x.shape
        crop_ratio = random.uniform(*self.crop_ratio)
        crop_timesteps = int(crop_ratio*W)
        start_idx = random.randint(0, W - crop_timesteps-1)
        x[:, start_idx:start_idx+crop_timesteps] = 0
        return x
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(crop_ratio={self.crop_ratio})"

    
class ChannelResize(object):
    """Scale amplitude of sample (per channel) by random factor in given magnitude range"""
    def __init__(self, range=(0.33, 3)):
        super().__init__()
        self.log_magnitude_range = torch.log(torch.tensor(range))        
    def __call__(self, x):
        C, W = x.shape
        resize_factors = torch.exp(torch.empty(C).uniform_(*self.log_magnitude_range))
        resize_factors_same_shape = resize_factors.repeat(W).reshape(x.shape)
        return resize_factors_same_shape * x
            
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(log_range={self.log_magnitude_range})"

class Reverse:
    def __call__(self, x):
        return torch.flip(x, dims=[-1])   
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

class Invert:
    def __call__(self, x):
        return torch.neg(x)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

class GaussianNoise:
    def __init__(self, scale=0.1) -> None:
        self.scale = scale
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """ Add gaussian noise of variance [self.scale] and mean 0 to the sample
        @param x (torch.FloatTensor of shape (b, c, w)): ecg sample
        @return noisy sample
        """
        noise = self.scale * torch.randn(x.shape, device=x.device)
        return x + noise

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.scale})"

class RandWanderer:
    def __init__(self, amp=(0.5,2), phase=(10, 100), gn_scale=0.05):
        super().__init__()
        self.amp = amp
        self.phase = phase
        self.gn_scale = gn_scale
        self.gn = GaussianNoise(gn_scale)

    def __call__(self, x):
        amp = torch.empty(1).uniform_(self.amp[0], self.amp[1]).item()
        sn = torch.linspace(self.phase[0], self.phase[1], x.shape[-1], device=x.device)
        sn = amp * torch.sin(math.pi * sn / 180)
        return x + sn + self.gn(x)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(amp={self.amp}, phase=({self.phase}), gn_scale={self.gn_scale})"
    

class BaselineWander:
    """Adds baseline wander to the sample.
    """
    def __init__(self, fs=250, Cmax=0.1, fc=0.5, fdelta=0.01, independent_channels=False):
        self.fs = fs
        self.Cmax = Cmax
        self.fc = fc
        self.fdelta = fdelta
        self.independent_channels = independent_channels

    def __call__(self, x):
        """@param x(torch.FloatTensor of shape (b, c, w))"""
        C, W = x.shape
        Crand = random.uniform(0,self.Cmax)
        noise = noise_baseline_wander(
            fs=self.fs, 
            N=W, 
            C=Crand, 
            fc=self.fc, 
            fdelta=self.fdelta,
            channels=C,
            independent_channels=self.independent_channels)
        return x + noise.t()

    def __str__(self):
        return "BaselineWander"


class PowerlineNoise(object):
    """Adds powerline noise to the sample.
    """
    def __init__(self, fs=250, Cmax=0.5, K=3):
        self.fs = fs
        self.Cmax = Cmax
        self.K = K
        
    def __call__(self, x):
        """@param x(torch.FloatTensor of shape (b, c, w))"""
        C, W = x.shape
        Crand = random.uniform(0, self.Cmax)
        noise =  noise_powerline(fs=self.fs, N=W, C=Crand, K=self.K, channels=C)
        return x + noise.t()

    def __str__(self):
        return "PowerlineNoise"


class EMNoise:
    """Adds electromyographic hf noise to the sample.
    """
    def __init__(self, Cmax=0.1, K=3):
        self.Cmax = Cmax
        self.K = K
    def __call__(self, x):
        """@param x(torch.FloatTensor of shape (b, c, w))"""
        C, W = x.shape
        Crand = random.uniform(0,self.Cmax)
        noise = noise_electromyographic(N=W, C=Crand, channels=C)
        return x + noise.t()


    def __str__(self):
        return "EMNoise"

class BaselineShift:
    """Adds abrupt baseline shifts to the sample.
    """
    def __init__(self, fs=250, Cmax=3., mean_segment_length=3, max_segments_per_second=0.3):
        self.fs = fs
        self.Cmax = Cmax
        self.mean_segment_length = mean_segment_length
        self.max_segments_per_second = max_segments_per_second

    def __call__(self, x):
        """@param x(torch.FloatTensor of shape (c, w))"""
        C, W = x.shape
        Crand= random.uniform(0,self.Cmax)
        noise = noise_baseline_shift(
            fs=self.fs, 
            N=W,
            C=Crand,
            mean_segment_length=self.mean_segment_length,
            max_segments_per_second=self.max_segments_per_second,
            channels=C
        )
        return x + noise.t()

    def __str__(self):
        return "BaselineShift"



def apply_filter(x, kernel):
    padding = len(kernel)//2
    kernel.unsqueeze_(0).unsqueeze_(0)
    C, W = x.shape
    x = x.unsqueeze(0)
    x = F.conv1d(x.reshape(1, 1, -1), kernel, padding=padding).view(1, C, W)
    return x.squeeze(0)
    


def noise_baseline_wander(fs=100, N=1000, C=1.0, fc=0.5, fdelta=0.01,channels=1,independent_channels=False,):
    '''baseline wander as in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5361052/
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale : 1)
    fc: cutoff frequency for the baseline wander (Hz)
    fdelta: lowest resolvable frequency (defaults to fs/N if None is passed)
    channels: number of output channels
    independent_channels: different channels with genuinely different outputs (but all components in phase) instead of just a global channel-wise rescaling
    '''
    if(fdelta is None):# 0.1
        fdelta = fs/N

    K = int((fc/fdelta)+0.5)
    t = torch.arange(0, N/fs, 1./fs).repeat(K).reshape(K, N)
    k = torch.arange(K).repeat(N).reshape(N, K).T
    phase_k = torch.empty(K).uniform_(0, 2*math.pi).repeat(N).reshape(N, K).T
    a_k = torch.empty(K).uniform_(0, 1).repeat(N).reshape(N, K).T
    pre_cos = 2*math.pi * k * fdelta * t + phase_k
    cos = torch.cos(pre_cos)
    weighted_cos = a_k * cos
    res = weighted_cos.sum(dim=0)
    return C*res
            
#     if(not(independent_channels) and channels>1):#just rescale channels by global factor
#         channel_gains = np.array([(2*random.randint(0,1)-1)*random.gauss(1,1) for _ in range(channels)])
#         signal = signal*channel_gains[None]
#     return signal

def noise_electromyographic(N=1000,C=1, channels=1):
    '''electromyographic (hf) noise inspired by https://ieeexplore.ieee.org/document/43620
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    channels: number of output channels
    '''
    #C *=0.3 #adjust default scale

    signal = torch.empty((N, channels)).normal_(0.0, C)
    
    return signal

def noise_powerline(fs=100, N=1000,C=1,fn=50.,K=3, channels=1):
    '''powerline noise inspired by https://ieeexplore.ieee.org/document/43620
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    fn: base frequency of powerline noise (Hz)
    K: number of higher harmonics to be considered
    channels: number of output channels (just rescaled by a global channel-dependent factor)
    '''
    #C *= 0.333 #adjust default scale
    t = torch.arange(0,N/fs,1./fs)
    
    signal = torch.zeros(N)
    phi1 = random.uniform(0,2*math.pi)
    for k in range(1,K+1):
        ak = random.uniform(0,1)
        signal += C*ak*torch.cos(2*math.pi*k*fn*t+phi1)
    signal = C*signal[:,None]
    if(channels>1):
        channel_gains = torch.empty(channels).uniform_(-1,1)
        signal = signal*channel_gains[None]
    return signal

def noise_baseline_shift(fs=100, N=1000,C=1.0,mean_segment_length=3,max_segments_per_second=0.3,channels=1):
    '''baseline shifts inspired by https://ieeexplore.ieee.org/document/43620
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    mean_segment_length: mean length of a shifted baseline segment (seconds)
    max_segments_per_second: maximum number of baseline shifts per second (to be multiplied with the length of the signal in seconds)
    '''
    #C *=0.5 #adjust default scale
    signal = torch.zeros(N)
    
    maxsegs = int((max_segments_per_second*N/fs)+0.5)
    
    for i in range(random.randint(0,maxsegs)):
        mid = random.randint(0,N-1)
        seglen = random.gauss(mean_segment_length,0.2*mean_segment_length)
        left = max(0,int(mid-0.5*fs*seglen))
        right = min(N-1,int(mid+0.5*fs*seglen))
        ak = random.uniform(-1,1)
        signal[left:right+1]=ak
    signal = C*signal[:,None]
    
    if(channels>1):
        channel_gains = 2*torch.randint(2, (channels,))-1 * torch.empty(channels).normal_(1, 1)
        signal = signal*channel_gains[None]
    return signal
