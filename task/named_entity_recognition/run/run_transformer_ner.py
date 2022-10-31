# -*- coding: utf-8 -*-
# @Author : zheng
# @Date : 2022/7/25 13:38

import os
import sys
import logging
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from named_entity_recognition.layers.pretrained_model import PretrainedModel
from named_entity_recognition.tools.data_preprocess import get_processor
from named_entity_recognition.tools.dataset import NerDataset, ner_collate_fn
from named_entity_recognition.tools.log_util import build_logger
from named_entity_recognition.models.transformer_ner import TransformerNER
from train import Trainer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
input_dir = os.path.join(parent_dir, 'dataset')
output_dir = os.path.join(parent_dir, 'output')
bert_dir = os.path.join(os.path.dirname(parent_dir), 'pretrained_models', 'save')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='transformer_ner', type=str, help='name of ner model')

    parser.add_argument('--input', default='dataset', type=str, help='folder of input data')
    parser.add_argument('--output', default='output', type=str, help='folder of output data, including logs, parameters, models')

    parser.add_argument('--dataset', default='ontonotes5', choices=['conll', 'clue', 'ontonotes4', 'ontonotes5'], type=str)

    parser.add_argument('--do_train', default=True, type=bool, help='whether to train a model from scratch')
    parser.add_argument('--epochs', default=30, type=int, help='num of training epochs')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size of training')
    parser.add_argument('--eval_batch_size', default=32, type=int, help='batch size of evaluating')
    parser.add_argument('--eval_per_epoch', default=1, type=int, help='how often evaluating the trained model on valid dataset during training')

    parser.add_argument('--bert_type', default='RoBERTa', choices=['BERT', 'RoBERTa', 'ALBERT', 'ELECTRA'], type=str)
    parser.add_argument('--bert_name', default='chinese-roberta-wwm-ext', choices=['chinese-bert-wwm-ext', 'chinese-roberta-wwm-ext'], type=str, help='bert name of the selected bert_type')
    parser.add_argument('--bert_cache_dir', default='', type=str, help='file location where the pretrained bert model is stored')

    parser.add_argument('--rnn', default=None, choices=['RNN', 'GRU', 'LSTM'], type=str)
    parser.add_argument('--crf', default=False, type=bool, help='whether to use CRF model')

    parser.add_argument('--use_gpu', default=True, type=bool, help='whether to train the model on GPU')
    parser.add_argument('--multi_gpu', default=False, type=bool, help='ensure multi-gpu training')
    parser.add_argument('--device_ids', default=[0], type=list, help='GPU index')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument('--learning_rate', default=5e-05)
    parser.add_argument('--crf_learning_rate', default=5e-05)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)

    parser.add_argument('--max_seq_length', default=512, type=int)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    if args.rnn is not None:
        args.model_name = f'NER_{args.bert_type}_{args.rnn}'
    else:
        args.model_name = f'NER_{args.bert_type}'
    if args.crf:
        args.model_name = args.model_name + '_CRF'
        if args.crf_learning_rate != args.learning_rate:
            args.model_name = args.model_name + f'_{args.crf_learning_rate}'

    args.input = os.path.join(input_dir, args.dataset)
    args.output = os.path.join(output_dir, args.model_name + '_' + args.dataset)
    args.bert_cache_dir = os.path.join(bert_dir, args.bert_name)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f'No such file or directory: {args.input}')
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(args.bert_cache_dir):
        os.makedirs(args.bert_cache_dir)

    if args.do_train:
        logger = build_logger(name=args.model_name, log_dir=args.output, type='train')
    else:
        logger = build_logger(name=args.model_name, log_dir=args.output, type='eval')

    logger.info(sys.argv)
    logger.info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    PM = PretrainedModel(task_type='lm', model_type=args.bert_type, model_name=args.bert_name, model_path=args.bert_cache_dir)
    config, tokenizer, encoder = PM.config, PM.tokenizer, PM.model

    logger.info('=' * 20 + 'Building dataset' + '=' * 20)

    ner_processor = get_processor(dataset=args.dataset)

    train_examples = ner_processor.get_examples(os.path.join(args.input, 'train.txt'), data_type='train')
    valid_examples = ner_processor.get_examples(os.path.join(args.input, 'valid.txt'), data_type='valid')
    test_examples = ner_processor.get_examples(os.path.join(args.input, 'test.txt'), data_type='test')

    label2id = {label: i for i, label in enumerate(ner_processor.get_labels(), 0)}

    train_dataset = NerDataset(examples=train_examples, label2id=label2id, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    valid_dataset = NerDataset(examples=valid_examples, label2id=label2id, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    test_dataset = NerDataset(examples=test_examples, label2id=label2id, tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=ner_collate_fn)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=ner_collate_fn)
    test_data_loader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=ner_collate_fn)

    logger.info(f'Complete building dataset, train/valid/test: {len(train_examples)}/{len(valid_examples)}/{len(test_examples)}')

    model = TransformerNER(enocder=encoder, rnn=args.rnn, crf=args.crf, hidden_dim=config.hidden_size, dropout=config.hidden_dropout_prob, tag_num=len(ner_processor.labels))

    train_steps = (int(len(train_examples) / args.train_batch_size) + 1) * args.epochs

    trainer = Trainer(cfg=args, model=model, label2id=label2id, train_steps=train_steps, basic_logger=logger)

    trainer.train(args, train_data_loader, valid_data_loader)

    # trainer.evaluate(test_data_loader, type='test')


if __name__ == '__main__':

    main()














