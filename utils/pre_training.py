from typing import Dict, Any

import torch
import wandb
import pandas as pd
from datasets import Dataset
from omegaconf import DictConfig
from audiomentations import Compose

from utils.train import train_model
from utils.model import MultilingualKws
from utils.utils import map_data_augmentation
from utils.data_generator import DataGenerator

device = ('cuda' if torch.cuda.is_available() else 'cpu')


def pre_training(
    cfg: DictConfig,
    label2id: Dict
) -> None:

    wandb_config = {
        **cfg.pretrain,
        **cfg.finetunning,
        **cfg.feature_extractor,
        **cfg.data_augmentation,
        **cfg.models,
        **cfg.data
    }

    run = wandb.init(
        project='Test_pretraining_multiplingual_kws',
        config=wandb_config
    )

    # Load a pandas dataframe
    train_df = pd.read_csv(cfg.data.train_csv_path)
    val_df = pd.read_csv(cfg.data.dev_csv_path)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    audio_augmentator = Compose([map_data_augmentation(aug_config) for aug_config in cfg.data_augmentation.audio_augmentation])

    train_dataset = DataGenerator(
        batch=train_dataset,
        sample_rate=cfg.feature_extractor.sample_rate,
        window_size=cfg.feature_extractor.window_size,
        hop_size=cfg.feature_extractor.hop_size,
        mel_bins=cfg.feature_extractor.mel_bins,
        period=cfg.feature_extractor.period,
        class_num=cfg.pretrain.classes_num,
        datatype='train',
        audio_augmentator=audio_augmentator,
        use_specaug=cfg.data_augmentation.spec_augment,
        freqm=cfg.data_augmentation.freqm,
        timem=cfg.data_augmentation.timem,
        label2id=label2id,
        path_column=cfg.data.path_column,
        label_column=cfg.data.label_column
    )

    val_dataset = DataGenerator(
        batch=val_dataset,
        sample_rate=cfg.feature_extractor.sample_rate,
        window_size=cfg.feature_extractor.window_size,
        hop_size=cfg.feature_extractor.hop_size,
        mel_bins=cfg.feature_extractor.mel_bins,
        period=cfg.feature_extractor.period,
        class_num=cfg.pretrain.classes_num,
        datatype='val',
        audio_augmentator=None,
        use_specaug=False,
        freqm=cfg.data_augmentation.freqm,
        timem=cfg.data_augmentation.timem,
        label2id=label2id,
        path_column=cfg.data.path_column,
        label_column=cfg.data.label_column
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.pretrain.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.pretrain.num_workers
    )

    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.pretrain.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.pretrain.num_workers
    )

    pre_training_model = MultilingualKws(
        classes_num=cfg.pretrain.classes_num,
        encoder_name=cfg.models.encoder_model_name,
        imagenet_pretrained=cfg.models.imagenet_pretrained
    )

    pre_training_model.to(device)

    train_model(
        train_dataloader=train_loader,
        val_dataloader=valid_loader,
        model=pre_training_model,
        embedding_layer=cfg.pretrain.embedding_layer,
        lr=cfg.pretrain.lr,
        warmup_rate=cfg.pretrain.warmup_rate,
        epochs=cfg.pretrain.epochs,
        weights_output_dir=cfg.pretrain.weights_output_dir,
        mixed_precision=cfg.pretrain.mixed_precision,
        early_stopping=cfg.pretrain.early_stopping,
   )

   # Finish a new run
    run.finish()