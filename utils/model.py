import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import formatter_single


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=formatter_single.FORMATTER)


def init_layer(layer):
    """Initialize a Linear or Convolutional layer"""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


class MultilingualKws(nn.Module):
    def __init__(
        self,
        classes_num: int,
        encoder_name: str,
        imagenet_pretrained: bool
    ):

        super(MultilingualKws, self).__init__()

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=imagenet_pretrained,
            in_chans=1,
            num_classes=0
        )

        self.fc1 = nn.Linear(1280, 2048, bias=True)
        self.fc2 = nn.Linear(2048, 2048, bias=True)
        self.fc3 = nn.Linear(2048, 1024, bias=True)

        self.fc_classifier = nn.Linear(1024, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)
        init_layer(self.fc_classifier)

    def forward(self, input):

        # Takes logits after layer of avg global pooling
        x = self.encoder(input)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        embeddings = F.selu(self.fc3(x))

        pre_training_output = torch.softmax(self.fc_classifier(embeddings), dim=-1)

        output_dict = {
            "embeddings": embeddings,
            "pre_training_output": pre_training_output
        }

        return output_dict


class TransferMultilingualKws(nn.Module):
    def __init__(
        self,
        pretraining_classes_num: int,
        encoder_name: str,
        freeze_base: bool,
        classes_num: int,
        pretrained_checkpoint_path: str
    ):
        """Classifier for a new task using pretrained MultilingualKws as a sub module."""
        super(TransferMultilingualKws, self).__init__()

        self.embedding_model = MultilingualKws(
            classes_num=pretraining_classes_num,
            encoder_name=encoder_name,
            imagenet_pretrained=False
        )

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(1024, classes_num, bias=True)

        self.init_weights()
        self.load_from_pretrain(pretrained_checkpoint_path=pretrained_checkpoint_path)

        if freeze_base:
            # Freeze Embedding pretrained layers
            log.info('Freezing Feature Extractor... ')
            for param in self.embedding_model.parameters():
                param.requires_grad = False


    def init_weights(self):
        init_layer(self.fc_transfer)


    def load_from_pretrain(self, pretrained_checkpoint_path: str):
        log.info('Loading Pretrained Weights from Embedding Model... ')
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.embedding_model.load_state_dict(checkpoint['model'])


    def forward(self, input):
        output_dict = self.embedding_model(input=input)

        embedding = output_dict['embeddings']

        tranfer_out =  torch.softmax(self.fc_transfer(embedding), dim=-1)
        output_dict['tranfer_out'] = tranfer_out

        return output_dict