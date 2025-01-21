import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import torchmetrics.functional.classification as Fc


def try_catch_fn(fn, default_val=np.nan):
    try:
        return fn()
    except Exception as e: 
        print('Error occured when evaluation metrics: ', e)
        return default_val
    

def metrics_headers(y_labels, categories, single=True, group=True):
    headers = []
    if single: 
        for label in y_labels:
            for m in ["auroc", "auprc", "f1score"]:
                headers += [f"{label}_{m}"]

    if group: 
        for category, elts in categories.items():
            idx = [y_labels.index(s) for s in elts if s in elts]
            num_labels = len(idx)
            if num_labels == 0: continue # No items in this category
            for m in ["auroc", "auprc", "f1score"]:
                for avg in ["micro", "macro"]:
                    headers += [f"{category}_{avg}_{m}"]
    return headers

def calculate_metrics(y_true, y_pred, device, y_labels, categories, single=True, group=True):
    y_pred = np.where(np.isnan(y_pred), 0, y_pred)
    y_pred_thresholded = (y_pred > 0.5).astype(int)
    t_y_true = torch.tensor(y_true, device=device, dtype=torch.long)
    t_y_pred = torch.tensor(y_pred, device=device)

    scores = {"tm": {}, "sk": {}}
    if single:
        for id, y_label in enumerate(y_labels):
            t_pred = t_y_pred[:, id]
            t_target = t_y_true[:, id]
            pred = y_pred[:, id]
            target = y_true[:, id]
            pred_threshold = y_pred_thresholded[:, id]
            scores["tm"][f"{y_label}_auroc"] = try_catch_fn(lambda: Fc.binary_auroc(t_pred, t_target))
            scores["sk"][f"{y_label}_auroc"] = try_catch_fn(lambda: roc_auc_score(target, pred))
            scores["tm"][f"{y_label}_auprc"] = try_catch_fn(lambda: Fc.binary_average_precision(t_pred, t_target))
            scores["sk"][f"{y_label}_auprc"] = try_catch_fn(lambda: average_precision_score(target, pred))
            scores["tm"][f"{y_label}_f1score"] = try_catch_fn(lambda: Fc.binary_f1_score(t_pred, t_target))
            scores["sk"][f"{y_label}_f1score"] = try_catch_fn(lambda: f1_score(target, pred_threshold,))
            
    if group:
        for category, elts in categories.items():
            idx = [y_labels.index(s) for s in elts if s in elts]
            num_labels = len(idx)
            if num_labels == 0: continue # No items in this category
            t_pred = t_y_pred[:, idx]
            t_target = t_y_true[:, idx]
            pred = y_pred[:, idx]
            target = y_true[:, idx]
            pred_threshold = y_pred_thresholded[:, idx]
            for avg in ["macro", "micro"]:
                scores["tm"][f"{category}_{avg}_auroc"] = \
                    try_catch_fn(lambda: Fc.multilabel_auroc(t_pred, t_target, num_labels, average=avg) if num_labels > 1 else Fc.binary_auroc(t_pred, t_target))
                scores["sk"][f"{category}_{avg}_auroc"] = \
                    try_catch_fn(lambda: roc_auc_score(target, pred, average=avg, ))
                scores["tm"][f"{category}_{avg}_auprc"] = \
                    try_catch_fn(lambda: Fc.multilabel_average_precision(t_pred, t_target, num_labels, average=avg) if num_labels > 1 else Fc.binary_average_precision(t_pred, t_target))
                scores["sk"][f"{category}_{avg}_auprc"] = \
                    try_catch_fn(lambda: average_precision_score(target, pred, average=avg, ))
                scores["tm"][f"{category}_{avg}_f1score"] = \
                    try_catch_fn(lambda: Fc.multilabel_f1_score(t_pred, t_target, num_labels, average=avg) if num_labels > 1 else Fc.binary_f1_score(t_pred, t_target))
                scores["sk"][f"{category}_{avg}_f1score"] = \
                    try_catch_fn(lambda: f1_score(target, pred_threshold, average=avg, ))
            
        
    return scores
                

