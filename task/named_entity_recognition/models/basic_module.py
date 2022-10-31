import os
import json
import time
import torch
import torch.nn as nn


class BasicModule(nn.Module):
    '''
    封装nn.Module, 提供 save 和 load 方法
    '''
    def __init__(self):
        super(BasicModule, self).__init__()


    def save_config(self, config, save_dir, filename='config.json'):
        """Save config into a directory.

        Args:
            config:
            save_dir: The directory to save config.
            filename: A file name for config.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        config_path = os.path.join(save_dir, filename)
        with open(config_path, 'w', encoding='utf-8') as out:
            json.dump(config, out, indent=4)


    def load_config(self, save_dir, filename='config.json'):
        """Load config from a directory.

        Args:
            save_dir: The directory to load config.
            filename: A file name for config.
        """
        pass


    def save_vocabs(self, save_dir, filename='vocabs.json'):
        """Save vocabularies to a directory.

        Args:
            save_dir: The directory to save vocabularies.
            filename:  The name for vocabularies.
        """
        pass

    def load_vocabs(self, save_dir, filename='vocabs.json'):
        """Load vocabularies from a directory.

        Args:
            save_dir: The directory to load vocabularies.
            filename:  The name for vocabularies.
        """
        pass


    def load_model(self, save_dir='', save_name='model.pth'):
        '''
        加载指定路径的模型
        '''
        pass


    def save_weights(self, save_dir='', save_name='model.pth'):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_path = os.path.join(save_dir, save_name)
        torch.save(self.state_dict(), model_save_path)


    def save(self, save_dir='', save_config=None, save_vocab=None):
        """Save this component to a directory.

        Args:
            save_dir: The directory to save this component.
        """
        if save_config is not None:
            self.save_config(save_config, save_dir)
        if save_vocab is not None:
            self.save_vocabs(save_dir)
        self.save_weights(save_dir)

    def load(self, **kwargs):
        """Load from a local/remote component.
        """
        pass




