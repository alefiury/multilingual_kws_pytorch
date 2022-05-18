import os
from dataclasses import dataclass
from collections import namedtuple
from typing import List

import torch
import torchaudio
import pandas as pd
import seaborn as sns
from audiomentations import Gain, AddGaussianNoise, PitchShift, AddBackgroundNoise, Shift

from .data_generator import DataGenerator


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass(frozen=True)
class Colors(metaclass=Singleton):
    BLACK: str = '\033[30m'
    RED: str = '\033[31m'
    GREEN: str = '\033[32m'
    YELLOW: str = '\033[33m'
    BLUE: str = '\033[34m'
    MAGENTA: str = '\033[35m'
    CYAN: str = '\033[36m'
    WHITE: str = '\033[37m'
    UNDERLINE: str = '\033[4m'
    RESET: str = '\033[0m'


@dataclass(frozen=True)
class LogFormatter(metaclass=Singleton):
    colors_single = Colors()
    TIME_DATA: str = colors_single.BLUE + '%(asctime)s' + colors_single.RESET
    MODULE_NAME: str = colors_single.CYAN + '%(module)s' + colors_single.RESET
    LEVEL_NAME: str = colors_single.GREEN + '%(levelname)s' + colors_single.RESET
    MESSAGE: str = colors_single.WHITE + '%(message)s' + colors_single.RESET
    FORMATTER = '['+TIME_DATA+']'+'['+MODULE_NAME+']'+'['+LEVEL_NAME+']'+' - '+MESSAGE


formatter_single = LogFormatter()
device = ('cuda' if torch.cuda.is_available() else 'cpu')
Info = namedtuple("Info", ["length", "sample_rate", "channels"])


def get_audio_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        # new version of torchaudio
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)


def map_data_augmentation(aug_config):
    aug_name = aug_config['name']
    del aug_config['name']
    if aug_name == 'additive':
        return AddBackgroundNoise(**aug_config)
    elif aug_name == 'gaussian':
        return AddGaussianNoise(**aug_config)
    elif aug_name == 'gain':
        return Gain(**aug_config)
    elif aug_name == 'pitch_shift':
        return PitchShift(**aug_config)
    elif aug_name == 'shift':
        return Shift(**aug_config)
    else:
        raise ValueError("The data augmentation '" + aug_name + "' doesn't exist !!")


def get_label_id(train_df_path, label_column):
    train_df = pd.read_csv(train_df_path)
    label2id, id2label = dict(), dict()

    labels = train_df[label_column].unique()
    labels.sort()

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    return label2id, id2label