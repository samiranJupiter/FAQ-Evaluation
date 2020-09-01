# import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import requests
import json
import os
import sklearn
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss
import warnings
import yaml
warnings.filterwarnings("ignore")


def normalized_hamming_loss(y_true, y_pred):
    """
    Returns hamming loss for given predicted and true labels
    """
    y_true = [[x.strip() for x in t] for t in y_true]
    mlb = MultiLabelBinarizer()
    y_true_binarized = mlb.fit_transform(y_true)
    y_pred_binarized = mlb.transform(y_pred)
    h_loss = hamming_loss(y_true_binarized, y_pred_binarized)
    label_density = np.mean([len(t) for t in y_true])
    h_loss = (h_loss * len(y_true_binarized[0]))/(2*label_density)
    print (len(y_true_binarized[0]))
    return h_loss  

def multi_label_precision(true, pred):
    """
    Returns multi-label precision for given true and predicted labels
    """
    if len(pred):
        precision = float(len(set(true).intersection(set(pred)))/len(pred))
    else:
        precision = 0
    return precision

def multi_label_recall(true, pred):
    """
    Returns multi-label recall for given true and predicted labels
    """
    if len(true):
        if len(true) <= len(pred):
            recall = float(len(set(true).intersection(set(pred)))/len(true))
        else:
            recall = float(len(set(true).intersection(set(pred)))/len(pred))
    else:
        recall = 0
    return recall

def multi_label_f1_score(true, pred):
    """
    Returns multi-label f1-score for given true and predicted labels
    """
    precision = multi_label_precision(true, pred)
    recall = multi_label_recall(true, pred)
    if precision or recall:
        f1_score = 2*(precision * recall)/(precision + recall)
    else:
        f1_score = 0
    return f1_score

def compute_multi_label_metrics(y_true, y_pred):
    """
    Returns precision, recall, f1 score and hamming loss for multi-label classification
    """
    metrics = {}
    precision_scores = []
    recall_scores = []
    f1_scores = []
    hamming_loss = 0
    for true, pred in zip(y_true, y_pred):
        #true = [x.strip() for x in true]
        precision = multi_label_precision(true, pred)
        recall = multi_label_recall(true, pred)
        f1_score = multi_label_f1_score(true, pred)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1_score)
    metrics['avg_precision'] = round(np.mean(precision_scores), 2)
    metrics['avg_recall'] = round(np.mean(recall_scores), 2)
    metrics['avg_f1'] = round(np.mean(f1_scores), 2)
    metrics['hamming_loss'] = round(normalized_hamming_loss(y_true, y_pred), 2)
    return metrics

def compute_single_label_metrics(y_true, y_pred):
    """
    Returns accuracy, precision, recall and f1 score for multi-class classification
    """
    metrics = {}
    metrics['accuracy'] = round(accuracy_score(y_true, y_pred), 2)
    metrics['precision'] = round(precision_score(y_true, y_pred, average="weighted"), 2)
    metrics['recall'] = round(recall_score(y_true, y_pred, average="weighted"), 2)
    metrics['f1'] = round(f1_score(y_true, y_pred, average="weighted"), 2) 
    return metrics