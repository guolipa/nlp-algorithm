# -*- coding: utf-8 -*-
# @Author : zheng
# @Date : 2022/7/20 12:45

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from .basic_module import BasicModule
from ..layers.pretrained_model import PretrainedModel
from ..layers.rnn import RNNModel
from ..layers.crf import CRF


class TransformerNER(BasicModule):

    def __init__(self, enocder=None, rnn=None, crf=False, hidden_dim=768, dropout=0.5, tag_num=None, extra_embedding=False) -> None:
        """
        A entity recognized tagging model use transformer（bert） as encoder.
        Args:
            encoder: A pretrained transformer.
            tag_num: Size of tagset.
            extra_embeddings: Extra embeddings which will be concatenated to the encoder outputs.
        """
        super().__init__()

        self.tag_num = tag_num
        self.encoder = enocder
        self.rnn = RNNModel(input_size=hidden_dim, hidden_size=hidden_dim, rnn_type=rnn) if rnn else None
        self.crf = CRF(num_tags=tag_num, batch_first=True) if crf else None
        self.classifier = nn.Linear(hidden_dim, tag_num)
        self.dropout = nn.Dropout(p=dropout)
        self.loss_fct = CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, label_mask=None, **kwargs):
        encoder_outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # [ batch_size * seq_length * hidden_dim ]
        sequence_output = encoder_outputs[0]
        if self.rnn:
            sequence_output, sequence_hn = self.rnn(sequence_output, attention_mask)
        # sequence_output = self.dropout(sequence_output)
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
            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)

        outputs = outputs + (logits, )

        return outputs











