# -*- coding: utf-8 -*-
# @Author : zheng
# @Date : 2022/7/25 13:50

import sys
import time
from tqdm import tqdm
from typing import List, Dict, Any, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AdamW, get_linear_schedule_with_warmup

from tools.metrics import ner_score_report

class Trainer(object):
    def __init__(self, cfg=None, model=None, label2id=None, train_steps=0, basic_logger=None):
        self.cfg = cfg
        self.model = model
        self.logger = basic_logger
        self.label2id = label2id
        self.id2label = {id: label for label, id in label2id.items()}
        self.out_dir = cfg.output

        self.bert_encoder = True if 'bert_type' in cfg else False

        if cfg.use_gpu and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{cfg.device_ids[0]}')
        else:
            self.device = torch.device('cpu')

        self.model.to(self.device)

        self.multi_gpu = False

        if cfg.use_gpu and cfg.multi_gpu and torch.cuda.device_count() > 0:
            self.multi_gpu = True
            torch.cuda.manual_seed_all(cfg.seed)
            self.model = nn.DataParallel(self.model.cuda(), device_ids=cfg.device_ids)
        if self.bert_encoder:
            self.optimizer, self.scheduler = self.build_transformer_optimizer(cfg, self.model, train_steps)
        else:
            self.optimizer = self.build_general_optimizer(cfg, self.model)
            self.scheduler = self.build_general_scheduler(cfg, self.optimizer)

    def train(self, cfg, train_loader, valid_loader):
        best_epoch = 0
        best_precision, best_recall, best_f1 = 0.0, 0.0, 0.0

        self.logger.info('=' * 20 + 'Start training' + '=' * 20)
        for epoch in range(cfg.epochs):
            self.model.train()
            total_loss = 0.0
            total_setps = 0
            for step, batch in tqdm(enumerate(train_loader), desc=f'Training epoch [{epoch}/{cfg.epochs}]',
                                    mininterval=0.5, colour='red', leave=False, file=sys.stdout):
                input_ids, input_mask, segment_ids, label_ids, label_mask = batch

                input_ids, input_mask, segment_ids, label_ids, label_mask = \
                    input_ids.to(self.device), input_mask.to(self.device), segment_ids.to(self.device), label_ids.to(self.device), label_mask.to(self.device)

                loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids, label_mask)

                if self.multi_gpu:
                    loss = loss.mean()
                if cfg.gradient_accumulation_steps > 1:
                    loss = loss / cfg.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)

                total_loss += loss.item()
                total_setps += 1

                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()

            self.logger.info('=' * 10 + 'Training epoch %d：step = %d, loss=%.5f' % (epoch, total_setps, total_loss / total_setps) + '=' * 10)

            if (epoch + 1) % cfg.eval_per_epoch == 0:
                self.model.eval()
                eval_ret = self.evaluate(valid_loader)
                if eval_ret['f1'] > best_f1:
                    best_epoch = epoch
                    best_precision = eval_ret['precision']
                    best_recall = eval_ret['recall']
                    best_f1 = eval_ret['f1']
                    self.logger.info('* Finding new best valid results, save model...')
                    self.model.save(self.out_dir, save_config=self.cfg.__dict__)
        self.logger.info('=' * 20 + ' End training ' + '=' * 20)
        self.logger.info('Best valid epoch: %d, best valid results: [ Precision: %.5f, Recall: %.5f, F1: %.5f ]' %
                         (best_epoch, best_precision, best_recall, best_f1))


    def evaluate(self, eval_loader: DataLoader=None, epoch=None, convert_label=True, type='valid') -> Dict[str, float]:
        label_true, label_pred = [], []

        total_loss = 0.0
        total_setps = 0

        start_time = time.time()

        for batch_id, eval_batch in tqdm(enumerate(eval_loader), desc=f'Evaluatining model',
                                    mininterval=0.5, colour='red', leave=False, file=sys.stdout):
            input_ids, input_mask, segment_ids, label_ids, label_mask = eval_batch
            input_ids, input_mask, segment_ids, label_ids, label_mask = \
                input_ids.to(self.device), input_mask.to(self.device), segment_ids.to(self.device), label_ids.to(self.device), label_mask.to(self.device)

            with torch.no_grad():
                loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids, label_mask)

            total_loss += loss
            total_setps += 1

            # print(label_mask.shape)
            # print(logits.shape)

            # [ batch_size * seq_length * tag_num ]
            # logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            if not self.cfg.crf:
                logits = logits.detach().cpu().numpy()

            label_ids = label_ids.to('cpu').numpy()
            label_mask = label_mask.to('cpu').numpy()


            for batch_index, seq_label in enumerate(logits):
                true_temp = []
                pred_temp = []
                for seq_index, label_id in enumerate(seq_label):
                    if self.bert_encoder and seq_index == 0:
                        continue
                    # if label_ids[batch_index][seq_index] == len(self.id2label)-1:
                    if label_mask[batch_index][seq_index] == 0 or (self.bert_encoder and label_ids[batch_index][seq_index] == len(self.id2label)-1):
                        label_true.append(true_temp)
                        label_pred.append(pred_temp)
                        # print(true_temp)
                        # print(pred_temp)
                        # print(label_mask[batch_index])
                        break
                    else:
                        true_temp.append(self.id2label[label_ids[batch_index][seq_index]])
                        pred_temp.append(self.id2label[logits[batch_index][seq_index]])

        # print(label_true)
        # print(label_pred)

        eval_ret = ner_score_report(label_true, label_pred)

        eval_ret['loss'] = total_loss / total_setps

        if epoch:
            self.logger.info('Evaluating epoch %d：used time = %f, loss = %.5f' % (epoch, time.time() - start_time, eval_ret['loss']))
        else:
            self.logger.info('Evaluating：used time = %f, loss = %.5f' % (time.time() - start_time, eval_ret['loss']))

        if type == 'test':
            self.logger.info('Evaluating trained model on test dataset...')

        self.logger.info('Correct_num: %d, Predict_num: %d, Gold_num: %d' % (eval_ret['correct_num'], eval_ret['pred_num'], eval_ret['gold_num']))
        self.logger.info('Precision: %.5f, Recall: %.5f, F1: %.5f' % (eval_ret['precision'], eval_ret['recall'], eval_ret['f1']))

        return eval_ret


    def predict(self, text: Union[str, List[str]]) -> Dict[str, str]:
        pass


    def build_transformer_optimizer(self, cfg, model, steps):
        param_optimizer = list(model.named_parameters())
        no_decay = ('bias', 'LayerNorm.bias', 'LayerNorm.weight')
        crf = ('crf', )
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay) and not any(nd in n for nd in crf)], 'weight_decay': cfg.weight_decay},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay) and not any(nd in n for nd in crf)], 'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in crf)], 'lr':cfg.crf_learning_rate, 'weight_decay':cfg.weight_decay}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate, correct_bias=True)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * cfg.warmup_proportion), steps)

        return optimizer, scheduler

    def build_general_optimizer(self, cfg, model):
        optimizer = None
        if cfg.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        return optimizer

    def build_general_scheduler(self, cfg, optimizer):
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.lr_factor, patience=cfg.lr_patience)
        # scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.8, total_iters=8)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=cfg.lr_factor)
        return scheduler










