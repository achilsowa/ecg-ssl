import torch
import torch.nn.functional as F

from src.utils.distributed import AllGather

SMALL_NUM=1e-7
LARGE_NUM = 1e9

def nt_xent_loss(Z_1_n, T_1_n, T_ids, off_diag=True, mutually_exclusive=True, normalize=True, temp=1.0, rank=0):
    """
    Here we try to match as possible the original SIMCLR TF implementation
    """
    if normalize:
        Z_1_n = F.normalize(Z_1_n)
        T_1_n = F.normalize(T_1_n)
    
    Z_1_n_large = AllGather.apply(Z_1_n)
    T_1_n_large = AllGather.apply(T_1_n)
    T_ids_large = AllGather.apply(T_ids)

    B = Z_1_n.shape[0]
    device = Z_1_n.device
    B_large = Z_1_n_large.shape[0]
    labels_idx = torch.arange(B, device=device) + rank * B
    labels = F.one_hot(labels_idx, B_large).float()

    # - We mask/exclude contribution of repetitive text before computing softmax (e.g see https://doi.org/10.1038/s44325-024-00010-0)
    # - (B x B_large) mask with 1 where a T_id is repeated in T_ids_large [outside the corresponding T_ids block]
    off_diag_masks = (T_ids.unsqueeze(1) == T_ids_large.unsqueeze(0)).int() - F.one_hot(labels_idx, B_large)  
    
    logits_ZT = torch.mm(Z_1_n, T_1_n_large.t()) / temp
    logits_TZ = torch.mm(T_1_n, Z_1_n_large.t()) / temp
    
    
    loss_a_off = loss_b_off = len_off_elts = 0
    if off_diag:
        def off_diag_loss():
            off_rows, off_cols = torch.where(off_diag_masks)
            labels_off = F.one_hot(off_cols, B_large).float()
            logits_ZT_off = logits_ZT[off_rows] # (len_off_diag, B_large)
            logits_TZ_off = logits_TZ[off_rows] 
            
            # - We mask/exclude contribution of repetitive text before softmax
            # - current off_diagnal is unmasked so it can contribute to softmax
            # - (len_off_diag, B_large)
            if mutually_exclusive:
                masks = (off_diag_masks + F.one_hot(labels_idx, B_large))[off_rows]
                masks[range(len(off_rows)), off_cols] = 0
                logits_ZT_off = logits_ZT_off - masks * LARGE_NUM  # (len_off_diag, B_large)
                logits_TZ_off = logits_TZ_off - masks * LARGE_NUM

            loss_a_off = F.cross_entropy(logits_ZT_off, labels_off)
            loss_b_off = F.cross_entropy(logits_TZ_off, labels_off)
            return loss_a_off, loss_b_off, len(off_rows)
        
        loss_a_off, loss_b_off, len_off_elts = off_diag_loss()
    
    if mutually_exclusive:
        logits_ZT = logits_ZT - off_diag_masks * LARGE_NUM 
        logits_TZ = logits_TZ - off_diag_masks * LARGE_NUM

    loss_a = F.cross_entropy(logits_ZT, labels)
    loss_b = F.cross_entropy(logits_TZ, labels)

    loss = loss_a + loss_b + loss_a_off + loss_b_off
    return loss, logits_ZT, labels, len_off_elts