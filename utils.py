from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import os
from os.path import join
from os import path


def evaluation_metrics(Y_true, Y_pred, split='test'):
    metrics = dict()
    metrics[split+'_accuracy'] = accuracy_score(Y_true, Y_pred)
    metrics[split+'_precision'] = precision_score(Y_true, Y_pred, average='macro')
    metrics[split+'_recall'] = recall_score(Y_true, Y_pred, average='macro')
    metrics[split+'_f1_score'] = f1_score(Y_true, Y_pred, average='macro')
    metrics[split+'_confusion_matrix'] = confusion_matrix(Y_true, Y_pred)

    return metrics


def save_model(model, name, prev_name):
    if prev_name is not None:
        if path.exists(prev_name):
            os.remove(prev_name)
    torch.save(model, name)


def root_and_binary_title(root, binary):
    if root:
        phrase_type = 'root'
    else:
        phrase_type = 'all'
    if binary:
        label = 'binary'
    else:
        label = 'fine'
    return phrase_type, label

def get_binary_label(sentiment):
    if sentiment < 2:
        return 0
    if sentiment > 2:
        return 1
    raise ValueError("Invalid sentiment")


def transformer_params(name):
    return {'batch_size': 32,
            'learning_rate': 1e-5}


