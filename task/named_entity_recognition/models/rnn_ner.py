# -*- coding: utf-8 -*-
# @Author : zheng
# @Date : 2022/7/20 13:05

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, NLLLoss

from .basic_module import BasicModule
from ..layers.pretrained_model import PretrainedModel
from ..layers.rnn import RNNModel
from ..layers.crf import CRF


class RNNNER(BasicModule):

    def __init__(self, rnn='LSTM', crf=False, vocab_size=0, embedding_dim=128, hidden_dim=128, tag_num=None,
                 pretrained_word_embedding=None, pretrained_char_embedding=None) -> None:
        """
        A entity recognized tagging model use transformer（bert） as encoder.
        Args:
            encoder: A pretrained transformer.
            tag_num: Size of tagset.
            extra_embeddings: Extra embeddings which will be concatenated to the encoder outputs.
        """
        super().__init__()
        self.tag_num = tag_num

        if pretrained_word_embedding is not None:
            embedding_dim = pretrained_word_embedding.embedding_dim
            self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
            self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_word_embedding.embeddings))
        else:
            # 不使用预训练的词向量
            self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        # if pretrained_char_embedding is not None:
        #     # pretrained_char_embedding 需要在 pretrained_word_embedding 基础上添加，如果只有pretrained_char_embedding，则将 pretrained_char_embedding 赋给 pretrained_word_embedding
        #     self.char_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=pretrained_char_embedding.embedding_dim, padding_idx=0)
        #     self.char_embedding.weight.data.copy_(torch.from_numpy(pretrained_char_embedding.embeddings))
        #     embedding_dim = embedding_dim + pretrained_char_embedding.embedding_dim

        hidden_dim = embedding_dim  # / 2 if embedding_dim > 100 else embedding_dim

        self.rnn = RNNModel(input_size=embedding_dim, hidden_size=hidden_dim, rnn_type=rnn)
        self.crf = CRF(num_tags=tag_num, batch_first=True) if crf else None
        self.classifier = nn.Linear(hidden_dim, tag_num)
        self.loss_fct = CrossEntropyLoss()
        # self.loss_fct = NLLLoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, label_mask=None):
        # [ batch_size * seq_length * hidden_dim ]
        input = self.word_embedding(input_ids)
        sequence_output, sequence_hn = self.rnn(input, attention_mask)
        # [ batch_size * seq_length * tag_num ]
        logits = self.classifier(sequence_output)

        outputs = ()
        if labels is not None:
            if self.crf is not None:
                loss = self.crf(emissions=logits, tags=labels, mask=label_mask.eq(1)) * (-1)
            else:
                if label_mask is not None:
                    active_loss = label_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.tag_num)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = self.loss_fct(active_logits, active_labels)
                else:
                    loss = self.loss_fct(logits.view(-1, self.tag_num), labels.view(-1))
            outputs = (loss, )

        if self.crf is not None:
            # [ batch_size * seq_length ]
            logits = self.crf.decode(emissions=logits, mask=label_mask.eq(1))
        else:
            # [ batch_size * seq_length ]
            # print(logits)
            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)

        # print(logits)

        outputs = outputs + (logits, )

        return outputs

