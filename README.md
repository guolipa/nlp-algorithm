# nlp-algorithm

## Named Entity Recognition
https://zhuanlan.zhihu.com/p/561776148
```
named_entity_recognition
│
├── dataset                                   # 数据文件夹
│   ├── clue                                  # clue数据集
│   ├── coll                                  # coll数据集
│   ├── ontonotes4                            # ontonotes4数据集 
│   ├── ontonotes5                            # ontonotes5数据集
│   │   ├── train.txt                         # 训练数据
│   │   ├── valid.txt                         # 验证数据
│   │   ├── test.txt                          # 测试数据
│  
├── layers
│   ├── crf.py                                # CRF
│   ├── rnn.py                                # RNN
│   ├── pretrained_model.py                   # BERT等预训练模型
│   ├── pretrained_embedding.py               # Word2vec等词向量
│
├── models                                    # 模型
│   ├── basic_module.py                       # 基础模型           
│   ├── rnn_ner.py                            # 基于RNN+CRF的命名实体识别模型
│   ├── transformer_ner.py                    # 基于Transformer+CRF的命名实体识别模型                                     
│
├── run
│   ├── run_rnn_ner.py                        # 训练RNN命名实体模型的主程序                  
│   ├── run_transformer_ner.py                # 训练Transformer命名实体识别模型的主程序
│   ├── train.py                              # 模型训练程序     
│ 
├── tools                                     # 基本工具
│   ├── data_preprocess.py                   # 数据加载预处理
│   ├── data_util.py                         # 数据处理
│   ├── dataset.py                           # 训练数据构建
│   ├── log_util.py                          # 日志记录
│   ├── metrics.py                           # 评价标准
```

```python
python ../run/run_transformer_ner.py
```