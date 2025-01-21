import torch
import torch.nn.functional as F

from src.utils.distributed import AllGather

SMALL_NUM=1e-7
LARGE_NUM = 1e9

def nt_xent_loss(Z, temp=1.0, rank=0):
    """
    Here we try to match as possible the original SIMCLR TF implementation
    """
    Z = F.normalize(Z)
    B = Z.shape[0]//2
    Z_1_n, Z_n1_2n = Z.split(B, 0)
  
    Z_1_n_large = AllGather.apply(Z_1_n)
    Z_n1_2n_large = AllGather.apply(Z_n1_2n)
    B_large = Z_1_n_large.shape[0]
    labels_idx = torch.arange(B, device=Z.device) + rank * B
    labels = F.one_hot(labels_idx, B_large * 2).float()
    masks = F.one_hot(labels_idx, B_large)
        
    logits_aa = torch.mm(Z_1_n, Z_1_n_large.t()) / temp
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.mm(Z_n1_2n, Z_n1_2n_large.t()) / temp
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.mm(Z_1_n, Z_n1_2n_large.t()) / temp
    logits_ba = torch.mm(Z_n1_2n, Z_1_n_large.t()) / temp

    loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], 1), labels)
    loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], 1), labels)
    loss = loss_a + loss_b

    return loss, logits_ab, labels