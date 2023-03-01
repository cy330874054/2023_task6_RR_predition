import models
import json
import sys

import torch
from transformers import BertTokenizer

import models
from eval import eval_model
from models import BertHSLN
from task import pubmed_task
from utils import get_device
from collections import Counter

def create_task(create_func):
    return create_func(train_batch_size=config["batch_size"], max_docs=MAX_DOCS)


def infer(model_paths, max_docs, prediction_output_json_path, device,num_model):
    ######### This function loads the model from given model path and predefined data. It then predicts the rhetorical roles and returns
    task = create_task(pubmed_task)
    model = getattr(models, config["model"])(config, [task]).to(device)
    folds = task.get_folds()
    test_batches = folds[0].test
    labels_dict = {}
    labels = []
    for path in model_paths:
        model.load_state_dict(torch.load(path))
        #metrics, confusion, labels_dict, class_report = eval_model(model, test_batches, device, task)
        labels_dict_model = {}
        model_eval = eval_model(model, test_batches, device, task)[2]
        labels_dict_model['doc_names'] = model_eval['doc_names']
        labels_dict_model['docwise_y_predicted'] = model_eval['docwise_y_predicted']
        labels.append(model_eval['docwise_y_predicted'])


    doc = labels[0]
    for i,sentence in enumerate(doc):
        for j in range(len(sentence)):
            sentence_predit_label = []
            for k in range(num_model):
                sentence_predit_label.append(labels[k][i][j])
            labels[0][i][j]=Counter(sentence_predit_label).most_common(1)[0][0]
    labels_dict['doc_names'] = labels_dict_model['doc_names']
    labels_dict['docwise_y_predicted'] = labels[0]
    return labels_dict

def write_in_hsln_format(input_json,hsln_format_txt_dirpath,tokenizer):

    #tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)
    json_format = json.load(open(input_json))
    final_string = ''
    filename_sent_boundries = {}
    for file in json_format:
        file_name=file['id']
        final_string = final_string + '###' + str(file_name) + "\n"
        filename_sent_boundries[file_name] = {"sentence_span": []}
        for annotation in file['annotations'][0]['result']:
            filename_sent_boundries[file_name]['sentence_span'].append([annotation['value']['start'],annotation['value']['end']])

            sentence_txt=annotation['value']['text']
            sentence_txt = sentence_txt.replace("\r", "")
            if sentence_txt.strip() != "":
                sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True, max_length=128)
                sent_tokens = [str(i) for i in sent_tokens]
                sent_tokens_txt = " ".join(sent_tokens)
                final_string = final_string + "NONE" + "\t" + sent_tokens_txt + "\n"
        final_string = final_string + "\n"

    with open(hsln_format_txt_dirpath + '/test_scibert.txt', "w+") as file:
        file.write(final_string)

    with open(hsln_format_txt_dirpath + '/train_scibert.txt', "w+") as file:
        file.write(final_string)
    with open(hsln_format_txt_dirpath + '/dev_scibert.txt', "w+") as file:
        file.write(final_string)
    with open(hsln_format_txt_dirpath + '/sentece_boundries.json', 'w+') as json_file:
        json.dump(filename_sent_boundries, json_file)

    return filename_sent_boundries

if __name__=="__main__":
    input_dir = './test.json'
    prediction_output_json_path = './predition.json'
    model_path = [
        'results/2023-01-02_15_34_56_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-23_20_26_09_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-23_19_41_35_pubmed-20k_baseline/0_0_model.pt',

        'results/2023-01-30_01_01_39_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-30_11_15_52_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-30_12_16_55_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-30_12_58_30_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-30_14_26_17_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-30_15_05_15_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-30_15_59_13_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-30_16_47_32_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-30_19_18_25_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-30_20_08_04_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-30_21_04_01_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-30_21_44_34_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-30_22_25_38_pubmed-20k_baseline/0_0_model.pt',

        'results/2023-01-31_12_09_22_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-31_12_59_17_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-31_13_25_54_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-31_14_09_13_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-31_14_51_19_pubmed-20k_baseline/0_0_model.pt',
        'results/2023-01-31_15_47_51_pubmed-20k_baseline/0_0_model.pt',

                  ]
    #[_,input_dir, prediction_output_json_path, model_path] = sys.argv

    BERT_VOCAB = "legalbert"
    BERT_MODEL = "legalbert"
    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)

    config = {
        "bert_model": BERT_MODEL,
        "bert_trainable": False,
        "model": BertHSLN.__name__,
        "cacheable_tasks": [],

        "dropout": 0.5,
        "word_lstm_hs": 768,
        "att_pooling_dim_ctx": 200,
        "att_pooling_num_ctx": 15,

        "lr": 3e-05,
        "lr_epoch_decay": 0.9,
        "batch_size": 32,
        "max_seq_length": 128,
        "max_epochs": 30,
        "early_stopping": 5,

    }


    MAX_DOCS = -1
    device = get_device(0)

    
    hsln_format_txt_dirpath ='datasets/pubmed-20k/'
    write_in_hsln_format(input_dir,hsln_format_txt_dirpath,tokenizer)
    filename_sent_boundries = json.load(open(hsln_format_txt_dirpath + '/sentece_boundries.json'))
    predictions = infer(model_path, MAX_DOCS, prediction_output_json_path, device,num_model=len(model_path))
    
    ##### write the output in format needed by revision script
    for doc_name,predicted_labels in zip(predictions['doc_names'],predictions['docwise_y_predicted']):
        filename_sent_boundries[doc_name]['pred_labels'] = predicted_labels
    with open(input_dir,'r') as f:
        input=json.load(f)
    pred_labels = predictions['docwise_y_predicted']
    for file in input:
        id=str(file['id'])
        pred_id=predictions['doc_names'].index(id)

        annotations=file['annotations']
        for i,label in enumerate(annotations[0]['result']):

            label['value']['labels']=[pred_labels[pred_id][i]]

    with open(prediction_output_json_path,'w') as file:
        json.dump(input,file)
