from sklearn import metrics
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

def accuracy(model, data, mask='train'):
    if mask == 'train':
        mask = data.train_mask
    elif mask == 'val':
        mask = data.val_mask
    elif mask == 'test':
        mask = data.test_mask
    else:
        mask = None
    model.eval()
    _, pred = model(data).max(dim=1)
    if mask is not None:
        correct = float(pred[mask].eq(data.y[mask]).sum().item())
        acc = correct / mask.sum().item()
    else:
        correct = float(pred.eq(data.y).sum().item())
        acc = correct / len(pred)
    return acc

def areaundercurve(model, data, mask='train', average="macro"):
    if mask == 'train':
        mask = data.train_mask
    elif mask == 'val':
        mask = data.val_mask
    else:
        mask = data.test_mask
    model.eval()
    _, pred = model(data).max(dim=1)

    y_true = data.y[mask].detach().cpu().numpy()
    y_pred = pred[mask].detach().cpu().numpy()
    
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)
    val = roc_auc_score(y_true, y_pred, average=average)
    return val

def average_precision_score_computation(model, data, mask='train'):
    if mask == 'train':
        mask = data.train_mask
    elif mask == 'val':
        mask = data.val_mask
    else:
        mask = data.test_mask
    model.eval()
    _, pred = model(data).max(dim=1)
#     correct = float(pred[mask].eq(data.y[mask]).sum().item())
#     acc = correct / mask.sum().item()
    y_true = data.y[mask].detach().cpu().numpy()
    y_pred = pred[mask].detach().cpu().numpy()
    ap = average_precision_score(y_true, y_pred)
    return ap