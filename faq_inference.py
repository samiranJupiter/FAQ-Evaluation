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
import datetime
warnings.filterwarnings("ignore")

'This script fetches predictions on the input data and saves those results'

CONFIG = "faq_config.yml"


def add_query_type(df):
    """
    Returns the input dataframe after adding column query type
    """
    query_type = []
    for i, row in df.iterrows():
        if len(row['true']) > 1:
            query_type.append("generic")
        else:
            query_type.append("specific")
    df['query_type'] = query_type
    return df

def preprocess_data(df):
    """
    Returns preprocessed test data
    """
    df['true'] = df['true'].apply(lambda x: [q.strip() for q in str(x).split(";")])
    df = add_query_type(df)
    return df

def get_payload(query, config):
    """
    Returns payload for the given config
    """
    payload = {
                "questions": [query],
                "top_k_reader": config['top_k_reader'],
                "score_weight": {
                                "es": config['es_weight'],
                                "embedding": config['embedding_weight']
                                },
                "es_query_options": {
                                "remove_stopwords": config['remove_stopwords'],
                                "search_in_question": config['search_in_question'],
                                "search_in_answer": config['search_in_answer']
                                }
                }
    return payload

def predict(url, payload):
    """
    Returns the response of the 
    """
    result = requests.post(url, data=json.dumps(payload))
    print(result.status_code)
    if result.status_code == 200:
        return result.json()
    else:
        return None

def parse_result(data):
    """
    Returns parsed response given raw response 
    """
    response = {'query': None,
               'result': []}
    result_keys = ['question', 'answer', 'score']
    result = []
    for answer in data['results'][0]['answers']:
        res_dict = {key: answer[key] for key in result_keys}
        result.append(res_dict)
    response['query'] = data['results'][0]['question']
    response['result'] = result
    return response

def extract_specific_results(df, config):
    """
    Returns results for specific queries
    """
    pred_questions = []
    # pred_answers = []
    pred_scores = []
    top_k_results = []
    count = 0
    for i, row in df.iterrows():
        print(row['query'])
        payload = get_payload(row['query'], config)
        result = predict(config['model_url'], payload)
        parsed_result = parse_result(result)
        pred_questions.append(parsed_result['result'][0]['question'])
        # pred_answers.append(parsed_result['result'][0]['answer'])
        pred_scores.append(parsed_result['result'][0]['score'])
        top_k_results.append(parsed_result['result'][0:config['top_k']])
        time.sleep(3)
    top_k_results = [[x['question'] for x in y] for y in top_k_results]
    df['pred'] = pred_questions
    # df['pred_answer'] = pred_answers
    df['confidence'] = pred_scores
    df['true'] = df['true'].apply(lambda x: x[0])
    df['top_'+str(config['top_k'])+'_pred'] = top_k_results
    return df

def extract_generic_results(df, config):
    """
    Returns results for generic queries
    """
    pred_questions = []
    count = 0
    for i, row in df.iterrows():
        print (row['query'])
        payload = get_payload(row['query'], config)
        result = predict(config['model_url'], payload)
        parsed_result = parse_result(result)
        pred_questions.append(parsed_result['result'][0:config['top_k']])
        time.sleep(3)
    pred_questions = [[x['question'] for x in y] for y in pred_questions]
    df['pred'] = pred_questions
    return df

def extract_results(df, config):
    """
    Returns results given config
    """
    specific_df = df[df['query_type'] == "specific"]
    generic_df = df[df['query_type'] == "generic"]
    specific_df = extract_specific_results(specific_df, config)
    generic_df = extract_generic_results(generic_df, config)
    return specific_df, generic_df

def get_timestamp():
    """
    Returns current timestamp
    """
    date_now = (str(datetime.datetime.now()))
    date = date_now.split(".")[0].split(" ")[0]
    time = date_now.split(".")[0].split(" ")[1]
    date = "".join(date.split("-"))
    time = "".join(time.split(":"))
    timestamp = date + time
    return timestamp


def load_config(filename):
    """
    Returns config file given filename
    """
    if os.path.exists(filename):
        with open(filename) as file:
            config = yaml.load(file, Loader=yaml.Loader)
        return config
    else:
        print("Config not found!")

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
    specific_results.to_csv(os.path.join(config['model_name'], config['results_dir'], specific_filename))
    generic_results.to_csv(os.path.join(config['model_name'], config['results_dir'], generic_filename))
    t2 = time.time()
    print ("Time elapsed:", round(float((t2-t1)/60), 2))

if __name__ == '__main__':
    main()