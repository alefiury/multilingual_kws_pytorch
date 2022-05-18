import logging
import argparse

import wandb
from omegaconf import OmegaConf

from utils.utils import get_label_id
from utils.utils import formatter_single
from utils.pre_training import pre_training
from utils.fine_tuning import fine_tuning

wandb.login()

# Logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=formatter_single.FORMATTER)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config_path',
        default='config/default.yaml',
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        '--pre_training',
        default=False,
        action='store_true',
        help='Trains model'
    )
    parser.add_argument(
        '--fine_tuning',
        default=False,
        action='store_true',
        help='Trains model'
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_path)

    label2id, _ = get_label_id(train_df_path=cfg.data.train_csv_path, label_column=cfg.data.label_column)

    if args.pre_training:
        pre_training(
            cfg=cfg,
            label2id=label2id
        )

    if args.fine_tuning:
        fine_tuning(
            cfg=cfg,
            label2id=label2id
        )

    else:
        print('Command was not given')

if __name__ == '__main__':
    main()