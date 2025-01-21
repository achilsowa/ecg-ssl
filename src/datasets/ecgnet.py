# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from logging import getLogger
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import neurokit2 as nk
from PIL import Image
import torch.utils

_GLOBAL_SEED = 0
logger = getLogger()

ECG_LENGTH = 2500
INT_MODE = "linear"

def make_ecgnet(
    batch_size,
    config,
    transform=None,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    drop_last=True,
):
    if isinstance(config, list):
        dataset_list = []
        for conf, tr in zip(config, transform):
            replicate = conf.get('replicate', 1)
            dataset_list += [EcgNet(conf, tr) for _ in range(replicate)]
        dataset = torch.utils.data.ConcatDataset(dataset_list)
    else:
        dataset = EcgNet(config, transform)
        
    logger.info(f'ECGNet dataset created, {len(dataset)}')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('ECGNet dataloader created')

    return dataset, data_loader, dist_sampler


class EcgNet(torch.utils.data.Dataset):
    def __init__(self, config, transform=None) -> None:
        super().__init__()
        self.x_max_len = config.get('x_max_len', np.inf)
        x_path = config['x_path']
        ext = x_path.split(".")[-1]
        if ext in ["parquet", "csv"] :
            self.strategy = PdStrategy(config, transform)
        elif ext == "npy":
            self.strategy = NpStrategy(config, transform)
        else:
            raise Exception(f"src_file extension should be in [csv|parquet|npy] not {ext}")
        self.error_count = 0
    
    def __getitem__(self, index):
        try:
            return self.strategy.getitem(index)
        except Exception as e:
            self.error_count += 1
            print(f'Error #{self.error_count} occured at: {index}:  {e}')
            # raise e
            return self.__getitem__(index-1)

    def __len__(self):
        return min(self.strategy.len, self.x_max_len)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(strategy={self.strategy}, len={self.__len__()})"
    

class PdStrategy:
    def __init__(self, config, transform=None):
        x_path = config.get('x_path')
        self.x_path = x_path
        self.x_label = config.get('x_label', 'ecg') 
        self.y_labels = config.get('y_labels', []) 
        self.use_diagnosis = config.get('use_diagnosis', False)
        self.use_diagnosis_id = config.get('use_diagnosis_id', False)
        self.clean = config.get('clean', False)
        self.resolution = config.get('resolution', ECG_LENGTH)
        self.transform = transform
        ext = x_path.split(".")[-1]
        self.ext = ext
        if ext == "parquet":
            self.data = pd.read_parquet(x_path)
            self.len = len(self.data)
        elif ext == "csv":
            self.data = pd.read_csv(x_path)
            self.len = len(self.data)
        self.has_png_path_col = 'png_path' in self.data.columns
        self.xml_prepend = config.get('xml_prepend', '/media/data1/ravram/DeepECG/ecg_png_parquet')

        # - Actually in MHI parquet, we have this typo. I modified it there so I do not have to 
        # - edit original parquet and risk breaking others' code
        if 'RV1 + SV6\xa0> 11 mm' in self.data.columns.tolist():
            self.data.rename(columns={'RV1 + SV6\xa0> 11 mm': 'RV1 + SV6 > 11 mm'}, inplace=True)
        self.data.reset_index(inplace=True, drop=True)

    def getitem(self, index):
        labels = self.data.loc[index, self.y_labels].tolist()
        labels = torch.tensor(labels)
        res = { "labels": labels, }
        if self.use_diagnosis: 
            res["text"] = self.data.loc[index, 'diagnosis']
        if self.use_diagnosis_id:
            res["text_id"] =  self.data.loc[index, 'diagnosis_id']

        if self.x_label == "img":
            if self.has_png_path_col:    
                png_path = self.data.loc[index, 'png_path']
            else:
                png_path = os.path.join(self.xml_prepend, self.data.loc[index, 'xml_path'].split('/')[-1] + '.png')
            x = Image.open(png_path)
            if self.transform is not None:
                x = self.transform(x)
            res["img"] = x
                
        else:
            npy_path = self.data.loc[index, 'npy_path']
            x = np.load(npy_path)
            clean_method = "neurokit" if self.clean else None
            x = npy_tensor(x, clean_method=clean_method, length=self.resolution, transform=self.transform)
            res["ecg"] =  x

        return res
        
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(xpath={self.x_path})"
    

class NpStrategy:
    def __init__(self, config, transform=None):
        x_path = config.get('x_path')
        y_path = config.get('y_path', None)
        self.x_path = x_path
        self.x_label = config.get('x_label', 'ecg') 
        self.y_path = y_path
        self.y_labels = config.get('y_labels', []) 
        self.all_labels = ['Sinusal','Regular','Monomorph','QS complex in V1-V2-V3','R complex in V5-V6','T wave inversion (inferior - II, III, aVF)','Left bundle branch block','RaVL > 11 mm','SV1 + RV5 or RV6 > 35 mm','T wave inversion (lateral -I, aVL, V5-V6)','T wave inversion (anterior - V3-V4)','Left axis deviation','Left ventricular hypertrophy','Bradycardia','Q wave (inferior - II, III, aVF)','Afib','Irregularly irregular','Atrial tachycardia (>= 100 BPM)','Nonspecific intraventricular conduction delay','Premature ventricular complex','Polymorph','T wave inversion (septal- V1-V2)','Right bundle branch block','Ventricular paced','ST elevation (anterior - V3-V4)','ST elevation (septal - V1-V2)','1st degree AV block','Premature atrial complex','Atrial flutter',"rSR' in V1-V2",'qRS in V5-V6-I, aVL','Left anterior fascicular block','Right axis deviation','2nd degree AV block - mobitz 1','ST depression (inferior - II, III, aVF)','Acute pericarditis','ST elevation (inferior - II, III, aVF)','Low voltage','Regularly irregular','Bifid','Junctional rhythm','Left atrial enlargement','ST elevation (lateral - I, aVL, V5-V6)','Atrial paced','Right ventricular hypertrophy','Delta wave','Wolff-Parkinson-White (Pre-excitation syndrome)','Prolonged QT','ST depression (anterior - V3-V4)','QRS complex negative in III','RaVL + SV3 > 28 mm (H) or 20 mm (F)','Q wave (lateral- I, aVL, V5-V6)','Hyperacute T wave (lateral, V5-V6)','Hyperacute T wave (septal, V1-V2)','Supraventricular tachycardia','ST downslopping','ST depression (lateral - I, avL, V5-V6)','2nd degree AV block - mobitz 2','U wave','ST depression et T inversion in V5 or V6','Large >0.08 s','R/S ratio in V1-V2 >1','RV1 + SV6 > 11 mm','Left posterior fascicular block','Right atrial enlargement','ST depression (septal- V1-V2)','Q wave (septal- V1-V2)','Q wave (anterior - V3-V4)','Hyperacute T wave (anterior, V3-V4)','ST upslopping','Right superior axis','Auricular bigeminy','Ventricular tachycardia','ST elevation (posterior - V7-V8-V9)','Ectopic atrial rhythm (< 100 BPM)','Lead misplacement','Biphasic','Ventricular bigeminy','J wave','Tall >2.5 mm','Third Degree AV Block','Sinus Pause','Acute MI','Early repolarization','Q wave (posterior - V7-V9)','Bi-atrial enlargement','LV pacing','Dextrocardia','Brugada','Ventricular Rhythm','ST depression (posterior - V7-V8-V9)','no_qrs']        
        self.y_indexes = [self.all_labels.index(y) for y in self.y_labels]
        self.transform = transform
        self.clean = config.get('clean', False)
        self.resolution = config.get('resolution', ECG_LENGTH)
        
        use_memmap, data = npy_load(data=None, filename=x_path)    
        self.xdata = None if use_memmap else data
        self.len = data.shape[0]
        
        if y_path is not None:
            use_memmap, ydata = npy_load(data=None, filename=y_path)
            self.ydata = None if use_memmap else ydata
        
        
    def getitem(self, index):
        if self.y_path is None:
            labels = []
        else:
            _, ydata = npy_load(data=self.ydata, filename=self.y_path)
            labels = ydata[index, self.y_indexes]
        labels = torch.tensor(labels)

        _, xdata = npy_load(data=self.xdata, filename=self.x_path)
        x = xdata[index]
        clean_method = "neurokit" if self.clean else None
        x = npy_tensor(x, clean_method=clean_method, length=self.resolution, transform=self.transform)
            
        return { "ecg": x, "labels": labels }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(xpath={self.x_path}, ypath={self.y_path})"       
    

def npy_tensor(x, clean_method=None, length=ECG_LENGTH, mode=INT_MODE, transform=None):
    """Utitily function to convert npy array to pytorch tensor. 
        Apply cleaning and transform if specified
    Here:
     1. we apply scaler if needed;
     2. we interpolate if needed;
     3. we remove nan;
     4. we reshape in (12, 2500), with type torch.float32

     param: x numpy array of shape (L, 12, 1) or (L, 12)
     return: torch tensor of shape (12, 2500)
    """
    x = x.squeeze().T
    x = np.where(np.isnan(x), 0, x)
    if clean_method is not None:
        x = np.apply_along_axis(
            # we expect 10 s duration ecg, so the sampling rate is len(ecg)//10
            lambda ecg: nk.ecg_clean(ecg, sampling_rate=len(ecg)//10, method=clean_method), 
            axis=1, 
            arr=x
        )
    x = torch.tensor(x, dtype=torch.float32)    
    x = F.interpolate(x.unsqueeze(0), length, mode=mode).squeeze(0)
    if transform is not None:
        x = transform(x)

    return x


def npy_load(data, filename,  threshold = 5*1024*1024*1024):
    """Utility to load small or large numpy array. 
    @param data: If specified, return that. 
    @param filename file location
    @param threshold value above which the memmap is used
    @return (use_memmap, data)"""
    if data is not None:
        return False, data
    
    fz = os.path.getsize(filename)
    use_memmap = fz > threshold
    return use_memmap, np.lib.format.open_memmap(filename, mode='r') if  use_memmap else np.load(filename)

