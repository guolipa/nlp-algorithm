# -*- coding: utf-8 -*-
# @Author : zheng
# @Date : 2022/4/21 11:09

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

from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer


LM_MODEL_CLASS = {
    'bert': (BertTokenizerFast, BertModel),
    'roberta': (RobertaTokenizerFast, RobertaModel),
    'albert': (AlbertTokenizerFast, AlbertModel),
    'electra': (ElectraTokenizerFast, ElectraModel),
    'auto': (AutoTokenizer, AutoModel)
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

# 预训练的模型存储的文件夹
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'save')
# 训练模型存储的文件夹
MODEL_OUT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')

logger = logging.getLogger(__name__)


class PretrainedModel(nn.Module):
    def __init__(self, config) -> None:
        super(PretrainedModel, self).__init__()
        self.config = config
        self.lang = config.lang
        self.task_type = config.task_type
        self.model_type = config.model_type
        self.model_name = config.model_name
        self.config.model_save_dir = os.path.join(MODEL_SAVE_DIR, self.model_name)
        self.config.model_out_dir = os.path.join(MODEL_OUT_DIR, self.model_name)

        self.model = None
        self.tokenizer = None

        # 加载预训练模型
        self.build_model()

    def build_model(self):
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
        task_class = TASK_MODEL_CLASS[self.task_type]
        if self.model_type not in task_class.keys():
            self.model_type = 'auto'
        tokenizer_class, model_class = task_class[self.model_type]
        model_save_path = self.config.model_save_dir
        if os.path.exists(model_save_path) and len(os.listdir(model_save_path)) > 0:
            self.tokenizer = tokenizer_class.from_pretrained(model_save_path)
            self.model = model_class.from_pretrained(model_save_path)
            logging.info(f'Load the pretrained model {self.model_name} from [{self.config.model_save_dir}]')
        else:
            self.tokenizer = tokenizer_class.from_pretrained(self.model_name)
            self.model = model_class.from_pretrained(self.model_name)
            logging.info(f'Download and load the pretrained model {self.model_name}')


    def encode(self, text: Union[str, List[str]]):
        """对输入的文本进行编码，输出embedding向量表示，分别针对短文本和长文本
            Bert支持的最大文本序列长度为512

            长文本处理策略：
                1. 长文本按512的长度切分成短句，k = L / 512
                    1) 取文本头部的512个字符、取尾部的512个字符、或去头部128字符加上尾部382字符作为输入
                    2）每句用BERT建模，最后取均值、最大、最小池化、attention操作；

        """
        # encode_input = { 'input_ids': [], 'token_type_ids': [], 'attention_mask': [] }
        encode_input = self.tokenizer(text)
        print(self.tokenizer.tokenize(text))


    def train_further(self):
        """
        加载Transformers库中提供的预训练模型，利用领域的文本数据对模型进行进一步的预训练
        """
        if self.config.train:
            # 构建数据加载器
            dataset, data_collator = self.build_dataloader()
            logging.info('Complete the loading and contruction of training data')
            # 构建训练器
            trainer = self.build_trainer(dataset, data_collator)
            logging.info('Complete the contruction of the model trainer')

            logging.info(f'{self.model}')

            logging.info('=' * 10 + 'Start training the model' + '=' * 10)
            start_time = time.time()
            trainer.train()
            end_time = time.time()
            logging.info('=' * 10 + 'End training the model' + '=' * 10)
            logging.info(f'The total time of model training：{(end_time - start_time) / 3600}')
            trainer.save_model(self.config.model_out_dir)
            logging.info(f'Save the trained model in {self.config.model_out_dir}')


    def build_dataloader(self):
        """
        读取训练数据，构建数据加载器和模型训练的数据处理器
        """

        # 读取数据集
        dataset = LineByLineTextDataset( tokenizer=self.tokenizer,
                                         file_path=self.config.data_save_dir,
                                         block_size=self.config.max_sequence_len
                                        )

        # 用来从训练数据中构建训练batch
        data_collator = DataCollatorForLanguageModeling( tokenizer=self.tokenizer,
                                                         mlm=self.config.mlm,
                                                         mlm_probability=self.config.mlm_probability
                                                        )

        return dataset, data_collator


    def build_trainer(self, dataset, data_collator):
        # 训练参数
        training_args = TrainingArguments( output_dir=self.config.model_out_dir,   # 输出路径，存储模型预测和checkpoints
                                           overwrite_output_dir=self.config.overwrite_output_dir,  # 覆盖输出路径的内容
                                           num_train_epochs=self.config.num_train_epochs,  # 训练的epoch
                                           per_device_train_batch_size=self.config.train_batch_size,
                                           save_strategy=self.config.save_strategy,
                                           no_cuda=self.config.no_cuda
                                          )

        # 训练器
        trainer = Trainer( model=self.model,
                           args=training_args,  # 训练参数
                           data_collator=data_collator,
                           tokenizer=self.tokenizer,
                           train_dataset=dataset  # 训练数据
                          )

        return trainer

    def segment_long_text(self, text, segment_len=500, overlap_len=50):
        segment_text = []
        if self.lang == 'zh':
            if len(text) <= segment_len:
                segment_text.append(text)
                return segment_text
            else:
                for num_segments in range(len(text) // segment_len):
                    if num_segments == 0:
                        text_piece = text[:segment_len]
                    else:
                        window = segment_len - overlap_len
                        text_piece = text[num_segments * window: num_segments * window + segment_len]
                        segment_text.append(text_piece)
                    return segment_text