# -*- coding: utf-8 -*-
# @Author : zheng
# @Date : 2022/7/20 13:18

import os
import time
import logging
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from transformers import \
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification, \
    BertTokenizerFast, BertModel, BertForSequenceClassification, BertForTokenClassification, \
    RobertaTokenizerFast, RobertaModel, RobertaForSequenceClassification, RobertaForTokenClassification, \
    AlbertTokenizerFast, AlbertModel, AlbertForSequenceClassification, AlbertForTokenClassification, \
    ElectraTokenizerFast, ElectraModel, ElectraForSequenceClassification, ElectraForTokenClassification

from transformers import BertConfig, RobertaConfig, AlbertConfig, ElectraConfig, AutoConfig

from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer


LM_MODEL_CLASS = {
    'bert': (BertConfig, BertTokenizerFast, BertModel),
    'roberta': (RobertaConfig, RobertaTokenizerFast, RobertaModel),
    'albert': (AlbertConfig, AlbertTokenizerFast, AlbertModel),
    'electra': (ElectraConfig, ElectraTokenizerFast, ElectraModel),
    'auto': (AutoConfig, AutoTokenizer, AutoModel)
}

SEQUENCE_MODEL_CLASS = {
    'bert': (BertTokenizerFast, BertForSequenceClassification),
    'roberta': (RobertaTokenizerFast, RobertaForSequenceClassification),
    'albert': (AlbertTokenizerFast, AlbertForSequenceClassification),
    'electra': (ElectraTokenizerFast, ElectraForSequenceClassification),
    'auto': (AutoTokenizer, AutoModelForSequenceClassification)
}

TOKEN_MODEL_CLASS = {
    'bert': (BertTokenizerFast, BertForTokenClassification),
    'roberta': (RobertaTokenizerFast, RobertaForTokenClassification),
    'albert': (AlbertTokenizerFast, AlbertForTokenClassification),
    'electra': (ElectraTokenizerFast, ElectraForTokenClassification),
    'auto': (AutoTokenizer, AutoModelForTokenClassification)
}

TASK_MODEL_CLASS = {
    'lm': LM_MODEL_CLASS,
    'seq_cls': SEQUENCE_MODEL_CLASS,
    'token_cls': TOKEN_MODEL_CLASS
}


class PretrainedModel(nn.Module):

    def __init__(self, task_type=None, model_type=None, model_name=None, model_path=None) -> None:
        super(PretrainedModel, self).__init__()
        self.task_type = task_type
        self.model_type = model_type
        self.model_name = model_name
        self.model_path = model_path

        self.config = None
        self.model = None
        self.tokenizer = None

        # load pretrained model
        _, _, _ = self.build_model(self.task_type, self.model_type, self.model_name, self.model_path)

    # @staticmethod
    def build_model(self, task_type, model_type, model_name, model_path):
        """
        根据配置参数config加载BERT模型 -> self.model
        * 中文模型
            - bert-base-chinese     谷歌提供的中文BERT模型
            - chinese-bert-wwm-ext    哈工大开源中文BERT模型 https://github.com/ymcui/Chinese-BERT-wwm
            - chinese-roberta-wwm-ext  哈工大开源中文Roberta模型

        * 英文模型
            - bert-base-uncased
            - roberta-base
            - electra-base-discriminator
        """
        task_class = TASK_MODEL_CLASS[task_type]
        if model_type not in task_class.keys():
            model_type = 'auto'
        config_class, tokenizer_class, model_class = task_class[model_type]
        model_save_path = model_path
        if not os.path.exists(model_save_path):
            os.mkdirs(model_save_path)
        if len(os.listdir(model_save_path)) > 0:
            self.config = config_class.from_pretrained(model_save_path)
            self.tokenizer = tokenizer_class.from_pretrained(model_save_path)
            self.model = model_class.from_pretrained(model_save_path)
            logging.info(f'Load the pretrained model {model_type}[{model_name}] from [{model_save_path}]')
        else:
            self.config = config_class.from_pretrained(model_name, cache_dir=model_save_path)
            self.tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=model_save_path)
            self.model = model_class.from_pretrained(model_name, cache_dir=model_save_path)
            logging.info(f'Download the pretrained model {model_type}[{model_name}] in [{model_save_path}]')

        return self.config, self.tokenizer, self.model


