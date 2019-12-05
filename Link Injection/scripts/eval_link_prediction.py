from sklearn import metrics
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer
import torch
import numpy as np

def accuracy(pred, true):
    if type(pred) == torch.Tensor:
        pred = pred.detach().cpu().numpy().ravel()
    if type(true) == torch.Tensor:
        true = true.detach().cpu().numpy().ravel()
    return np.mean(pred == true)

def auc_roc(pred, true, average="macro"):
    if type(pred) == torch.Tensor:
        pred = pred.detach().cpu().numpy().ravel()
    if type(true) == torch.Tensor:
        true = true.detach().cpu().numpy().ravel()
    lb = LabelBinarizer()
    lb.fit(true)
    y_true = lb.transform(true)
    y_pred = lb.transform(pred)
    val = roc_auc_score(true, pred, average=average)
    return val

def ap(pred, true):
    if type(pred) == torch.Tensor:
        pred = pred.detach().cpu().numpy().ravel()
    if type(true) == torch.Tensor:
        true = true.detach().cpu().numpy().ravel()
    ap_score = average_precision_score(true, pred)
    return ap_score

def precision(pred, true):
    if type(pred) == torch.Tensor:
        pred = pred.detach().cpu().numpy().ravel()
    if type(true) == torch.Tensor:
        true = true.detach().cpu().numpy().ravel()
    p = precision_score(true, pred)
    return p

def recall(pred, true):
    if type(pred) == torch.Tensor:
        pred = pred.detach().cpu().numpy().ravel()
    if type(true) == torch.Tensor:
        true = true.detach().cpu().numpy().ravel()
    r = recall_score(true, pred)
    return r