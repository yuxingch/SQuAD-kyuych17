""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
# import argparse
import json
import sys
import os
import glob
from evaluate import f1_score, metric_max_over_ground_truths

def initialize_keys(dataset):
    lst = []
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                lst.append(qa['id'])
    return lst


def prediction_to_dictionary(prev_dict, predictions, keys):
    for curr_id in keys:
        pred = predictions[curr_id]       
        if curr_id in prev_dict:
            prev_dict[curr_id].append(pred)
        else:
            prev_dict[curr_id] = [pred]
    return prev_dict


def ensemble_strategy(dictionary, dataset):
    result = {}
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                # if qa['id'] not in predictions:
                #     message = 'Unanswered question ' + qa['id'] + \
                #               ' will receive score 0.'
                #     print(message, file=sys.stderr)
                #     continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction_list = dictionary[qa['id']]
                f1 = 0-100000
                for p in prediction_list:
                    curr_f1 = metric_max_over_ground_truths(f1_score, p, ground_truths)
                    if curr_f1 > f1:
                        f1 = curr_f1
                        result[qa['id']] = p
    return result


if __name__ == '__main__':
    # expected_version = '1.1'
    # parser = argparse.ArgumentParser(
    #     description='Evaluation for SQuAD ' + expected_version)
    # # TODO: load several prediction files
    # parser.add_argument('dataset_file', help='Dataset file')
    # # parser.add_argument('prediction_file', help='Prediction File')
    # parser.add_argument('prediction_path', help='The directory that stores predtions')
    # args = parser.parse_args()
    
    dataset_file = './data/tiny-dev.json'

    with open(dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        # if (dataset_json['version'] != expected_version):
        #     print('Evaluation expects v-' + expected_version +
        #           ', but got dataset with v-' + dataset_json['version'],
        #           file=sys.stderr)
        dataset = dataset_json['data']
    # with open(args.prediction_file) as prediction_file:
    #     predictions = json.load(prediction_file)
    keys = initialize_keys(dataset)
    # print (keys)
    candidate_dictionary = {}
    
    path = './multi_preds/'
    json_files = [path+pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
    print(json_files)
    for i in range(len(json_files)):
        with open(json_files[i]) as prediction_file:
            predictions = json.load(prediction_file)
            candidate_dictionary = prediction_to_dictionary(candidate_dictionary, predictions, keys)

    result = ensemble_strategy(candidate_dictionary, dataset)
    with open('test.json', 'w') as fp:
        json.dump(result, fp)