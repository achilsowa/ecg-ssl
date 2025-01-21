# ECG-SSL

Pytorch codebase for implementing some ssl methods on ECGs.

The structure of this project is based on [ijepa](https://github.com/facebookresearch/ijepa) by Meta

We implemented several contrastive learning methods for ECGs:

## JEPA
We adapt [ijepa](https://github.com/facebookresearch/ijepa) to 1-D mask instead of original 2-D masks for images. We also implement some transforms, specific to ECGs

## SIMCLR
We adapt [simclr](https://github.com/google-research/simclr) in pytorch and for 1-D signals

## BYOL
We implement a simple BYOL loss function

## MECLIP
MECLIP stands for Mutually exclusive CLIP. This methods is usable when we have both ECGs and diagnostic texts. Most of the time, some diagnostic are repetitive so we end up having N_distint_diagnosis << N_distinct_ecgs. This methods apply classic CLIP learning, but when two distinct ECGs have same diagnostic, we can handle them differently:
- When 02 ECGs have common diagnostic pair, no care, we just apply classic CLIP with all other pair in the batch considered negative pairs
- When 02 ECGs have common diagnostic pair, we exclude the off diagonal from negative pairs
- When 02 ECGs have common diagnostic pair, we exclude the off diagonal from negative pair and we count it as another positive pair

## Code Structure

```
.
├── configs                   # directory in which all experiment '.yaml' configs are stored
├── src                       # the package
│   ├── train.py              #   the I-JEPA training loop
│   ├── helper.py             #   helper functions for init of models & opt/loading checkpoint
│   ├── transforms.py         #   pre-train data transforms
│   ├── datasets              #   datasets, data loaders, ...
│   ├── models                #   model definitions
│   ├── masks                 #   mask collators, masking utilities, ... (useful for jepa)
│   └── utils                 #   shared utilities
├── ssls
│   ├── byol                  #   train for byol
│   ├── jepa                  #   train for jepa
│   ├── simclr                #   simclr train and contrastive loss
│   ├── meclip                #   meclip train and contrastive loss
│   
├── main_distributed.py       # entrypoint for launch distributed I-JEPA pretraining on SLURM cluster
└── main.py                   # entrypoint for launch I-JEPA pretraining locally on your machine
```

**Config files:**
Note that all experiment parameters are specified in config files (as opposed to command-line-arguments). See the [configs/](configs/) directory for example config files.


## Launching ECG-SSL pretraining

### Single-GPU training
This implementation starts from the [ssls/main.py](ssls/main.py), which parses the experiment config file and runs the pre-training locally on a multi-GPU (or single-GPU) machine. For example, to run MECLIP pretraining on GPUs "0","1", and "2" on a local machine using the config [configs/meclip.yaml](configs/meclip.yaml), type the command:
```
python -m ssls.main \
  --fname configs/meclip.yaml \
  --devices cuda:0 cuda:1 cuda:2
```

## Launching ECG-SSL finetuning

### Single-GPU training
This implementation starts from the [evals/main.py](evals/main.py), which parses the experiment config file and runs the fine-tuning locally on a multi-GPU (or single-GPU) machine. It can start either from a pre-trained model either from random initialization. For example, to run ECG-SSL finetuning on GPU "0" on a local machine using the config,  [configs/evals-meclip.yaml](configs/evals-meclip.yaml), which will load the weight of MECLIP pre-trained model, type the command:
```
python -m evals.main \
  --fname configs/evals-meclip.yaml \
  --devices cuda:0
```
---

### Requirements
* Python 3.8 (or newer)
* PyTorch 2.0
* torchvision
* Other dependencies: pyyaml, numpy, opencv, submitit

