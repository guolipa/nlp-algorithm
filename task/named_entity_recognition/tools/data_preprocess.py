# -*- coding: utf-8 -*-
# @Author : zheng
# @Date : 2022/7/22 11:14

import os
from typing import List, Set
from abc import ABC, abstractmethod

from .dataset import InputExample, InputFeatures


class NerProcessor(ABC):
    """Basic class for data converters for token-level sequence classification data sets."""

    @abstractmethod
    def get_examples(self, data_dir, data_type='train'):
        """Gets a collection of `InputExample`s for the train/valid/test set."""
        pass

    @abstractmethod
    def get_labels(self):
        """Gets the list of labels for this data set."""
        pass

    @abstractmethod
    def _create_examples(self, lines, data_type):
        """Creats a collection of `InputExample`s for the train/valid/test set."""
        pass

    @abstractmethod
    def _read_file(self, input_file):
        """Reads raw data from the original data file"""
        pass


class ConllNERProcessor(NerProcessor):
    def __init__(self):
        self.labels = set()

    def get_examples(self, data_path, data_type='train'):
        return self._create_examples(
            self._read_file(data_path), data_type
        )

    def get_labels(self):
        additional_labels = ['X', '[CLS]', '[SEP]']
        self.labels = list(self.labels)
        self.labels.extend(additional_labels)
        print(self.labels)
        return self.labels

    def _create_examples(self, data, data_type):
        examples = []
        for i, (sentence, label) in enumerate(data):
            uid = '%s-%s' % (data_type, i)
            text = ' '.join(sentence)
            label = label
            examples.append(InputExample(uid, text=text, label=label))
        return examples

    def _read_file(self, input_file):
        data = []
        sentence = []
        label = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == '\n':
                    if len(sentence) > 0:
                        data.append((sentence, label))
                        sentence = []
                        label = []
                    continue
                token_label = line.split(' ')
                sentence.append(token_label[0])
                label.append(token_label[-1][:-1])
                self.labels.add(token_label[-1][:-1])

            if len(sentence) > 0:
                data.append((sentence, label))

        return data


class Ontonotes5Processor(NerProcessor):
    def __init__(self):
        self.labels = set()

    def get_examples(self, data_path, data_type='train'):
        return self._create_examples(
            self._read_file(data_path), data_type
        )

    def get_labels(self):
        additional_labels = ['X', '[CLS]', '[SEP]']
        self.labels = list(self.labels)
        self.labels.extend(additional_labels)
        return self.labels

    def _create_examples(self, data, data_type):
        examples = []
        for i, (sentence, label) in enumerate(data):
            uid = '%s-%s' % (data_type, i)
            text = ' '.join(sentence)
            label = label
            examples.append(InputExample(uid, text=text, label=label))
        return examples

    def _read_file(self, input_file):
        data = []
        sentence = []
        label = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) == 0 or line[0] == '\n' or line.startswith('-DOCSTART'):
                    if len(sentence) > 0:
                        data.append((sentence, label))
                        sentence = []
                        label = []
                    continue
                token_label = line.split('\t')
                sentence.append(token_label[0])
                label.append(token_label[-1][:-1])
                self.labels.add(token_label[-1][:-1])

            if len(sentence) > 0:
                data.append((sentence, label))

        return data


class ClueProcessor(NerProcessor):
    def __init__(self):
        self.labels = set()

    def get_examples(self, data_path, data_type='train'):
        return self._create_examples(
            self._read_file(data_path), data_type
        )

    def get_labels(self):
        additional_labels = ['X', '[CLS]', '[SEP]']
        self.labels = list(self.labels)
        self.labels.extend(additional_labels)
        return self.labels

    def _create_examples(self, data, data_type):
        examples = []
        for i, (sentence, label) in enumerate(data):
            uid = '%s-%s' % (data_type, i)
            text = ' '.join(sentence)
            label = label
            examples.append(InputExample(uid, text=text, label=label))
        return examples

    def _read_file(self, input_file):
        data = []
        sentence = []
        label = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) == 0 or line[0] == '\n':
                    if len(sentence) > 0:
                        data.append((sentence, label))
                        sentence = []
                        label = []
                    continue
                token_label = line.split('\t')
                sentence.append(token_label[0])
                label.append(token_label[-1][:-1])
                self.labels.add(token_label[-1][:-1])

            if len(sentence) > 0:
                data.append((sentence, label))

        return data


def get_processor(dataset):
    if dataset == 'conll':
        return ConllNERProcessor()
    elif dataset == 'ontonotes4':
        return Ontonotes5Processor()
    elif dataset == 'ontonotes5':
        return Ontonotes5Processor()
    elif dataset == 'clue':
        return ClueProcessor()



