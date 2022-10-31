# -*- coding: utf-8 -*-
# @Author : zheng
# @Date : 2022/8/23 16:37

import os
import sys
import logging
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from named_entity_recognition.tools.data_preprocess import get_processor
from named_entity_recognition.tools.dataset import NerDataset, ner_collate_fn, build_vocab
from named_entity_recognition.tools.log_util import build_logger
from named_entity_recognition.models.rnn_ner import RNNNER
from named_entity_recognition.layers.pretrained_embedding import PretrainedEmbedding
from train import Trainer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
input_dir = os.path.join(parent_dir, 'dataset')
output_dir = os.path.join(parent_dir, 'output')
pretrained_dir = os.path.join(os.path.dirname(parent_dir), 'pretrained_models', 'save')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='rnn_ner', type=str, help='name of ner model')

    parser.add_argument('--input', default='dataset', type=str, help='folder of input data')
    parser.add_argument('--output', default='output', type=str, help='folder of output data, including logs, parameters, models')

    parser.add_argument('--dataset', default='clue', choices=['conll', 'clue', 'ontonotes4', 'ontonotes5'], type=str)

    parser.add_argument('--do_train', default=True, type=bool, help='whether to train a model from scratch')
    parser.add_argument('--epochs', default=30, type=int, help='num of training epochs')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size of training')
    parser.add_argument('--eval_batch_size', default=32, type=int, help='batch size of evaluating')
    parser.add_argument('--eval_per_epoch', default=1, type=int, help='how often evaluating the trained model on valid dataset during training')

    parser.add_argument('--word_embedding', default=False, type=bool)
    parser.add_argument('--word_type', default='word2cev', choices=['word2cev', 'glove', 'fasttext'], type=str, help='bert name of the selected bert_type')
    parser.add_argument('--word_cache_dir', default='', type=str, help='file location where the pretrained embeddings are stored')
    # parser.add_argument('--char_embedding', default=False, type=bool)
    # parser.add_argument('--char_cache_dir', default='', type=str, help='file location where the pretrained embeddings are stored')
    parser.add_argument('--max_vocab_size', default=50000, type=int)
    parser.add_argument('--embedding_dim', default=256, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)

    parser.add_argument('--rnn', default='LSTM', choices=['RNN', 'GRU', 'LSTM'], type=str)
    parser.add_argument('--crf', default=False, type=bool, help='whether to use CRF model')

    parser.add_argument('--use_gpu', default=True, type=bool, help='whether to train the model on GPU')
    parser.add_argument('--multi_gpu', default=False, type=bool, help='ensure multi-gpu training')
    parser.add_argument('--device_ids', default=[0], type=list, help='GPU index')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    #Precision: 0.76694, Recall: 0.71076, F1: 0.73778

    parser.add_argument('--optimizer', default='Adam', choices=['SGD', 'Adagrad', 'RMSprop', 'Adam', 'AdamW'])
    parser.add_argument('--scheduler', default='LinearLR', choices=[''])
    parser.add_argument('--learning_rate', default=1e-3)
    parser.add_argument('--lr_factor', default=0.9, help='学习率的衰减率')
    parser.add_argument('--lr_patience', default=3, help='学习率衰减的等待epoch')
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)

    parser.add_argument('--max_seq_length', default=512, type=int)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    args.model_name = f'NER_{args.rnn}'
    if args.crf:
        args.model_name = f'NER_{args.rnn}_CRF'

    args.input = os.path.join(input_dir, args.dataset)
    args.output = os.path.join(output_dir, args.model_name + '_' + args.dataset)
    # args.word_cache_dir = os.path.join(pretrained_dir, 'ctp/ctb.50d.vec')
    # args.char_cache_dir = os.path.join(pretrained_dir, 'ctp/uni.ite50.vec')
    # skip_first_line = False
    args.word_cache_dir = os.path.join(pretrained_dir, 'wikipedia_zh/sgns.wiki.word-char')
    args.char_cache_dir = os.path.join(pretrained_dir, 'wikipedia_zh/sgns.wiki.bigram-char')
    skip_first_line = True
    if not os.path.exists(args.input):
        raise FileNotFoundError(f'No such file or directory: {args.input}')
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.do_train:
        logger = build_logger(name=args.model_name, log_dir=args.output, type='train')
    else:
        logger = build_logger(name=args.model_name, log_dir=args.output, type='eval')

    logger.info(sys.argv)
    logger.info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info('=' * 20 + 'Building dataset' + '=' * 20)

    ner_processor = get_processor(dataset=args.dataset)

    train_examples = ner_processor.get_examples(os.path.join(args.input, 'train.txt'), data_type='train')
    valid_examples = ner_processor.get_examples(os.path.join(args.input, 'valid.txt'), data_type='valid')
    test_examples = ner_processor.get_examples(os.path.join(args.input, 'test.txt'), data_type='test')

    label2id = {label: i for i, label in enumerate(ner_processor.get_labels(), 0)}
    print(label2id)
    # {'I-PRODUCT': 0, 'I-ORG': 1, 'I-EVENT': 2, 'I-WORK_OF_ART': 3, 'B-QUANTITY': 4, 'I-ORDINAL': 5, 'I-LOC': 6,
    #  'B-PERSON': 7, 'I-DATE': 8, 'I-TIME': 9, 'I-LANGUAGE': 10, 'I-NORP': 11, 'B-ORDINAL': 12, 'B-LAW': 13,
    #  'B-CARDINAL': 14, 'O': 15, 'B-FAC': 16, 'B-DATE': 17, 'B-GPE': 18, 'I-PERCENT': 19, 'I-MONEY': 20, 'I-LAW': 21,
    #  'B-PRODUCT': 22, 'I-PERSON': 23, 'I-FAC': 24, 'I-GPE': 25, 'B-ORG': 26, 'B-WORK_OF_ART': 27, 'B-LOC': 28,
    #  'I-QUANTITY': 29, 'B-NORP': 30, 'I-CARDINAL': 31, 'B-TIME': 32, 'B-LANGUAGE': 33, 'B-PERCENT': 34, 'B-MONEY': 35,
    #  'B-EVENT': 36, 'X': 37, '[CLS]': 38, '[SEP]': 39}

    vocab = build_vocab(train_examples)

    train_dataset = NerDataset(examples=train_examples, label2id=label2id, vocab=vocab, max_seq_length=args.max_seq_length)
    valid_dataset = NerDataset(examples=valid_examples, label2id=label2id, vocab=vocab, max_seq_length=args.max_seq_length)
    test_dataset = NerDataset(examples=test_examples, label2id=label2id, vocab=vocab, max_seq_length=args.max_seq_length)

    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=ner_collate_fn)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=ner_collate_fn)
    test_data_loader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=ner_collate_fn)

    examples = [train_examples, valid_examples, test_examples]
    data_loaders = [train_data_loader, valid_data_loader, test_data_loader]

    logger.info(f'Complete building dataset, train/valid/test: {len(train_examples)}/{len(valid_examples)}/{len(test_examples)}')

    pm_word = None
    if args.word_embedding:
        pm_word = PretrainedEmbedding(vocab=vocab.get_stoi())
        pm_word.build_embeddings_from_file(pretrained_file=args.word_cache_dir, skip_first_line=skip_first_line)
    # if args.char_embedding:
    #     pm_char = PretrainedEmbedding(vocab=vocab.get_stoi())
    #     pm_char.build_embeddings_from_file(pretrained_file=args.char_cache_dir, skip_first_line=False)

    model = RNNNER(rnn=args.rnn, crf=args.crf, vocab_size=len(vocab), embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
                   tag_num=len(label2id), pretrained_word_embedding=pm_word)

    train_steps = (int(len(train_examples) / args.train_batch_size) + 1) * args.epochs

    trainer = Trainer(cfg=args, model=model, label2id=label2id, train_steps=train_steps, basic_logger=logger)

    trainer.train(args, train_data_loader, valid_data_loader)

    # trainer.evaluate(test_data_loader, type='test')

















