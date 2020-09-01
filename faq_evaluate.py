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
import ast
import datetime
from datetime import date
import metrics
from metrics import *
import faq_inference
from faq_inference import *
warnings.filterwarnings("ignore")

CONFIG = "faq_config.yml"

'This script evaluates the model results'

def get_top_n_predictions(df, n, k):
    """
    Returns top n predictions 
    """
    y_true = list(df['true'])
    y_pred = list(df['top_'+str(k)+'_pred'])
    y_pred = [ast.literal_eval(json.dumps(x)) for x in y_pred]
    y_pred = [[str(x) for x in y] for y in y_pred]
    y_pred = [x[0:n] for x in y_pred]
    #print (y_pred)
    y_pred_top_n = []
    for true, pred in zip(y_true, y_pred):
        if true in pred:
            y_pred_top_n.append(true)
        else:
            y_pred_top_n.append(pred[0])
    return y_pred_top_n

def evaluate_specific(df):
    """
    Returns single label classification metrics for top prediction
    """
    y_true = list(df['true'])
    y_pred = list(df['pred'])
    metrics = compute_single_label_metrics(y_true, y_pred)
    return metrics


def evaluate_specific_top_n(df, n, config):
    """
    Returns single label classification metrics for top n predictions
    """
    y_true = list(df['true'])
    y_pred = get_top_n_predictions(df, n, config['top_k'])
    metrics = compute_single_label_metrics(y_true, y_pred)
    return metrics

def evaluate_generic(df):
    """
    Returns multi label classicaion metrics
    """
    y_true = list(df['true'])
    y_pred = list(df['pred'])
    metrics = compute_multi_label_metrics(y_true, y_pred)
    return metrics

def save_results(df, config, top_pred_metrics, top_2_metrics, top_3_metrics, generic_metrics ):
    row = []
    row.append(date.today())
    test_set = config['test_set'].split("/")[1]
    row.append(test_set)
    row.append(config['model_name'])
    row.append(top_pred_metrics['accuracy'])
    row.append(top_2_metrics['accuracy'])
    row.append(top_3_metrics['accuracy'])
    row.append(generic_metrics['avg_precision'])
    row.append(generic_metrics['avg_recall'])
    row.append(generic_metrics['avg_f1'])
    row.append(generic_metrics['hamming_loss'])
    df.loc[len(df.index)] = row
    return df

def initialize_results():
    columns = ['date', 'test_set', 'model', 'top_acc', 'top_2_acc', 'top_3_acc', \
             'avg_precision', 'avg_recall', 'avg_f1', 'hamming_loss']
    df = pd.DataFrame(columns=columns)
    return df

def main():
    t1 = time.time()
    config = load_config(CONFIG)
    test_set = config['test_set']
    test_df = pd.read_csv(test_set)
    test_df = preprocess_data(test_df)
    timestamp = get_timestamp()
    if not os.path.exists(config['model_name']):
        os.mkdir(config['model_name'])
    if not os.path.exists(os.path.join(config['model_name'], config['results_dir'])):
        os.mkdir(os.path.join(config['model_name'], config['results_dir']))
    specific_results, generic_results = extract_results(test_df, config)
    # save results
    specific_filename = "specific_results_" + timestamp + ".csv"
    generic_filename = "generic_results_" + timestamp + ".csv"
    specific_results.to_csv(os.path.join(config['model_name'], config['results_dir'], specific_filename), index=None)
    generic_results.to_csv(os.path.join(config['model_name'], config['results_dir'], generic_filename), index=None)
    top_pred_metrics = evaluate_specific(specific_results)
    top_2_metrics = evaluate_specific_top_n(specific_results, 2, config)
    top_3_metrics = evaluate_specific_top_n(specific_results, 3, config)
    generic_metrics = evaluate_generic(generic_results)
    print(":::::::::::::Performance on specific queries:::::::::::::::")
    print("\nPredicted label: Top predicted result")
    print(top_pred_metrics)
    print("\nPredicted label: Either of the 2 predicted results")
    print(top_2_metrics)
    print("\nPredicted label: Either of the 3 predicted results")
    print(top_3_metrics)
    print("\n::::::::::::::::Performance on generic queries:::::::::::::::")
    print(generic_metrics)
    #save_results(config, top_pred_metrics, top_2_metrics, top_3_metrics, generic_metrics)
    t2 = time.time()
    print ("Time elapsed:", round(float((t2-t1)/60), 2))

if __name__ == '__main__':
    main()


    
