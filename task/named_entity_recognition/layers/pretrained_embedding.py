# -*- coding: utf-8 -*-
# @Author : zheng
# @Date : 2022/8/30 15:21

import os
import numpy as np
from typing import Dict, Any

class PretrainedEmbedding():
    def __init__(self, embedding_num: int=None, embedding_dim: int=None, embeddings: Any=None, vocab: Dict=None):
        self.embedding_num = embedding_num
        self.embedding_dim = embedding_dim
        self.embeddings = embeddings
        self.vocab = vocab   # token ---> id
        if not self.embedding_num and vocab:
            self.embedding_num = len(vocab)
        if not self.embeddings and (self.embedding_num and self.embedding_dim):
            self.embeddings = np.zeros([self.embedding_num, self.embedding_dim])

    def build_embeddings_from_file(self, pretrained_file=None, skip_first_line=False, separator=' ', re_norm=False):
        pretrained_embeddings, pretrained_dim, _ = self.load_pretrained_embeddings(pretrained_file, skip_first_line, separator)
        self.embedding_dim = pretrained_dim
        if not self.embeddings:
            self.embeddings = np.zeros([self.embedding_num, self.embedding_dim])

        match = 0
        bound = np.sqrt(1.0 / self.embedding_dim)
        for token, token_id in self.vocab.items():
            if token in pretrained_embeddings:
                if re_norm:
                    self.embeddings[token_id, :] = self.embedding_norm(pretrained_embeddings[token])
                else:
                    self.embeddings[token_id, :] = pretrained_embeddings[token]
                match += 1
            else:
                self.embeddings[token_id, :] = np.random.uniform(-bound, bound, self.embedding_dim)

        print(f'vocab size: {self.embedding_num}, match pretrained: {match} | {match * 100.0 / self.embedding_num}')

    def load_pretrained_embeddings(self, pretrained_file=None, skip_first_line=False, separator=' '):
        if not os.path.exists(pretrained_file):
            raise FileNotFoundError(f'No such file or directory:{pretrained_file}')

        pretrained_size = 0
        pretrained_dim = -1
        pretrained_embeddings = dict()

        with open(pretrained_file, 'r', encoding='utf-8') as fr:
            for i, line in enumerate(fr):
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                contents = line.split(separator)
                if skip_first_line and i == 0:
                    pretrained_size, pretrained_dim = int(contents[0]), int(contents[1])
                    continue
                if not skip_first_line:
                    pretrained_size += 1
                tokens = contents[0]
                features = contents[1:]
                if pretrained_dim < 0:
                    pretrained_dim = len(features)
                else:
                    assert pretrained_dim == len(features)
                pretrained_embeddings[tokens] = np.array(features, dtype=np.float)

        return pretrained_embeddings, pretrained_dim, pretrained_size

    def embedding_norm(self, embedding):
        '''
        max-min（min-max normalization、range scaling）、Mean normalization、Standardization (Z-score Normalization)、Scaling to unit length（L1、L2）
        :return:
        '''
        norm = np.sqrt(np.sum(np.square(embedding)))
        norm_embedding = embedding / norm
        return norm_embedding


if __name__ == '__main__':
    file = '/home/zcw/project_python/nlp-algorithm/task/pretrained_models/save/ctp/uni.ite50.vec'

    PE = PretrainedEmbedding()
    embed, dim, size = PE.load_pretrained_embeddings(pretrained_file=file)

    print(dim)
    print(size)
    print(embed['中'])


