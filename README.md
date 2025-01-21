# ECG-SSL

Pytorch codebase for implementing some contrastive learning methods on ECGs.

The structure of this project is based on ![ijepa](https://github.com/facebookresearch/ijepa) by Meta

We implement several contrastive learning methodes for ECGs:

## JEPA
We adapt [ijepa](https://github.com/facebookresearch/ijepa) to 1-D mask instead of original 2-D masks for images. We also implement some transforms, specific to ECGs

## SIMCLR
We adapt [simclr]() in pytorch and for 1-D signals

## BYOL
We implement a simple BYOL loss function

## MECLIP
MECLIP stands for Mutually exclusive CLIP. This methods is usable when we have both ECGs and diagnostic texts. Most of the time, some diagnostic are repetitive so we end up having N_distint_diagnosis << N_distinct_ecgs. This methods apply classic CLIP learning, but when two distinct ECGs have same diagnostic, they are considered as being par of the same positive pair. In practive we have 3 differents possibility:
- When 02 ECGs have two diagnostic pair, we exlude it from the loss of the other when evaluating the contrastive loss
- When 02 
- classic CLIP where we do not care if two disctinct ECGs have same diagnostic text


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
│   ├── masks                 #   mask collators, masking utilities, ...
│   └── utils                 #   shared utilities
├── ssls
│   ├── byol                  #   train and loss for bYOL Contrastive loss
│   ├── helper.py             #   helper functions for init of models & opt/loading checkpoint
│   ├── transforms.py         #   pre-train data transforms
│   ├── datasets              #   datasets, data loaders, ...
│   
├── main_distributed.py       # entrypoint for launch distributed I-JEPA pretraining on SLURM cluster
└── main.py                   # entrypoint for launch I-JEPA pretraining locally on your machine
```

**Config files:**
Note that all experiment parameters are specified in config files (as opposed to command-line-arguments). See the [configs/](configs/) directory for example config files.

## Launching I-JEPA pretraining

### Single-GPU training
This implementation starts from the [main.py](main.py), which parses the experiment config file and runs the pre-training locally on a multi-GPU (or single-GPU) machine. For example, to run I-JEPA pretraining on GPUs "0","1", and "2" on a local machine using the config [configs/in1k_vith14_ep300.yaml](configs/in1k_vith14_ep300.yaml), type the command:
```
python main.py \
  --fname configs/in1k_vith14_ep300.yaml \
  --devices cuda:0 cuda:1 cuda:2
```
*Note: This example is just used for illustrative purposes, as the ViT-H/14 config should be run on 16 A100 80G GPUs for an effective batch-size of 2048, in order to reproduce our results.*

### Multi-GPU training
In the multi-GPU setting, the implementation starts from [main_distributed.py](main_distributed.py), which, in addition to parsing the config file, also allows for specifying details about distributed training. For distributed training, we use the popular open-source [submitit](https://github.com/facebookincubator/submitit) tool and provide examples for a SLURM cluster.

For example, to pre-train on 16 A100 80G GPUs using the pre-training experiment configs specificed inside [configs/in1k_vith14_ep300.yaml](configs/in1k_vith14_ep300.yaml), type the command:
```
python main_distributed.py \
  --fname configs/in1k_vith14_ep300.yaml \
  --folder $path_to_save_submitit_logs \
  --partition $slurm_partition \
  --nodes 2 --tasks-per-node 8 \
  --time 1000
```

---

### Requirements
* Python 3.8 (or newer)
* PyTorch 2.0
* torchvision
* Other dependencies: pyyaml, numpy, opencv, submitit

