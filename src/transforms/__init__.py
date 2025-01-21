from src.utils.helper import get_norm_from_pkl

def make_group_transforms(type, config):
    if isinstance(config, list):
        return [make_transforms(type, **conf) for conf in config]
    else:
        return make_transforms(type, **config)

        
def make_transforms(
    type = "ecg",
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    random_crop=True,
    center_crop=False,
    horizontal_flip=False,
    color_distortion=False,
    gray_scale=True,
    gaussian_blur=0.5,
    gaussian_noise=False,
    sobel_derivative=False,
    reverse=False,
    invert=False,
    rand_wanderer=False,
    baseline_wanderer=False,
    baseline_shift=False,
    em_noise=False,
    pl_noise=False,
    time_out=False,
    scale=False,    
    normalization=None,
    **kwargs
):
    if isinstance(normalization, str):
        normalization = get_norm_from_pkl(normalization)
    if type == "ecg":
        from src.transforms.ecg import make_transforms as make_transforms_ecg
        return make_transforms_ecg(
            crop_size=crop_size,
            crop_scale=crop_scale,
            random_crop=random_crop,
            gaussian_blur=gaussian_blur,
            normalization=normalization,
            gaussian_noise=gaussian_noise,
            sobel_derivative=sobel_derivative,
            reverse=reverse,
            invert=invert,
            rand_wanderer=rand_wanderer,
            baseline_wanderer=baseline_wanderer,
            baseline_shift=baseline_shift,
            em_noise=em_noise,
            pl_noise=pl_noise,
            time_out=time_out,
            scale=scale,    
        )
    elif type == "img":
        from src.transforms.img import make_transforms as make_transforms_img
        return make_transforms_img(
            crop_size=crop_size,
            crop_scale=crop_scale,
            color_jitter=color_jitter,
            random_crop=random_crop,
            center_crop=center_crop,
            horizontal_flip=horizontal_flip,
            color_distortion=color_distortion,
            gaussian_blur=gaussian_blur,
            gray_scale=gray_scale,
            normalization=normalization
        )
    

def make_eval_transforms(
    type = "ecg",
    resolution=2500,
    gray_scale=True,
    normalization=[[0.,], [1.,]]
):
    if isinstance(normalization, str):
        normalization = get_norm_from_pkl(normalization)
    
    if type == "ecg":
        from src.transforms.ecg import make_eval_transforms as make_transforms_ecg
        return make_transforms_ecg(normalization=normalization)
    elif type == "img":
        from src.transforms.img import make_eval_transforms as make_transforms_img
        return make_transforms_img(
            resolution=resolution,
            gray_scale=gray_scale,
            normalization=normalization
        )