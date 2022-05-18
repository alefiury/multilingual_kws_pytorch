import os
from typing import Dict, Optional, Any

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import tensorflow as tf
from tensorflow.lite.experimental.microfrontend.python.ops import (
    audio_microfrontend_op as frontend_op,
)

class DataGenerator(Dataset):
    def __init__(
        self,
        batch: Dict,
        sample_rate: int,
        window_size: int,
        hop_size: int,
        mel_bins: int,
        period: int,
        class_num: int,
        datatype: str,
        audio_augmentator: Optional[Any],
        use_specaug: bool,
        freqm: Optional[int],
        timem: Optional[int],
        label2id: dict,
        path_column: str,
        label_column: str
    ):

        'Initialization'
        self.batch = batch

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.period = period

        self.class_num = class_num

        self.datatype = datatype

        self.audio_augmentator = audio_augmentator
        self.use_specaug = use_specaug
        self.freqm = freqm
        self.timem = timem

        self.label2id = label2id

        self.path_column = path_column
        self.label_column = label_column

    def __len__(self):
        return self.batch.num_rows


    def cutorpad(self, audio: np.ndarray) -> np.ndarray:
        """
        Cut or pad an audio
        """
        effective_length = self.sample_rate * self.period
        len_audio = audio.shape[0]

        # If audio length is less than wished audio length
        if len_audio < effective_length:
            new_audio = np.zeros(effective_length)
            new_audio[:len_audio] = audio
            audio = new_audio

        # If audio length is bigger than wished audio length
        elif len_audio > effective_length:
            audio = audio[:effective_length]

        # If audio length is equal to wished audio length
        else:
            audio = audio

        # Expand one dimension related to the channel dimension
        return audio

    def to_micro_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        sample_rate = self.sample_rate
        window_size_ms = (self.window_size * 1000) / sample_rate
        window_step_ms = (self.hop_size * 1000) / sample_rate
        int16_input = tf.cast(tf.multiply(audio, 32768), tf.int16)
        # https://git.io/Jkuux
        micro_frontend = frontend_op.audio_microfrontend(
            int16_input,
            sample_rate=sample_rate,
            window_size=window_size_ms,
            window_step=window_step_ms,
            num_channels=self.mel_bins,
            out_scale=1,
            out_type=tf.float32,
        )
        output = tf.multiply(micro_frontend, (10.0 / 256.0)).numpy()
        return output

    def __getitem__(self, index: int) -> Dict[np.ndarray, np.ndarray]:

        audio_path = self.batch[index][self.path_column]
        audio_original, sr = librosa.load(audio_path, sr=None)
        # Audiomentations takes float32
        audio = self.cutorpad(audio_original).astype(np.float32)

        if self.audio_augmentator is not None and self.datatype == 'train':
            audio = self.audio_augmentator(audio, sample_rate=self.sample_rate)

        mel_S = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.mel_bins,
            n_fft=self.window_size,
            hop_length=self.hop_size
        )

        mel_S_db = librosa.power_to_db(mel_S, ref=np.max)
        mel_S_db = np.expand_dims(mel_S_db, axis=0)
        # Conversion to use SpecAugment (numpy -> Torch Tensor)
        mel_S_db = torch.from_numpy(mel_S_db)
        # SpecAugmentation
        if self.use_specaug and self.datatype=='train':
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
            if self.freqm != 0:
                mel_S_db = freqm(mel_S_db)
            if self.timem != 0:
                mel_S_db = timem(mel_S_db)

        label = int(self.label2id[self.batch[index][self.label_column]])

        return {
            'image': mel_S_db,
            'target': label
        }