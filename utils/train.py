import os
import logging
from typing import Union, Optional

import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

from utils.model import MultilingualKws, TransferMultilingualKws
from utils.scores import model_accuracy, model_f1_score

device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Logger
log = logging.getLogger(__name__)


def train_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: Union[MultilingualKws, TransferMultilingualKws],
    embedding_layer: str,
    lr: float = 1e-3,
    warmup_rate: float = 0.2,
    epochs: int = 120,
    weights_output_dir: str = 'weights',
    mixed_precision: bool = False,
    early_stopping: bool = False,
    patience: Optional[bool] = None
) -> None:

    # Create directory to save weights if it does not exist
    os.makedirs(weights_output_dir, exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # Save gradients of the weights
    wandb.watch(
        model,
        criterion,
        log='all',
        log_freq=10
    )

    loss_min = np.Inf

    # Initialize early stopping
    current_patience = 0

    # Initialize scheduler
    num_train_steps = int(len(train_dataloader) * epochs)
    num_warmup_steps = int(warmup_rate * epochs * len(train_dataloader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps
    )

    # Initialize mixed precision
    scaler = GradScaler(enabled=mixed_precision)

    model.train()
    for e in tqdm(range(epochs)):
        running_loss = 0
        train_accuracy = 0
        train_f1_score = 0
        for train_batch_count, sample in enumerate(tqdm(train_dataloader)):
            train_audio, train_label = sample['image'].to(device), sample['target'].to(device)

            optimizer.zero_grad()

            with autocast(enabled=mixed_precision):
                out = model(train_audio)
                # Model outputs a dict
                out = out[embedding_layer]
                train_loss = criterion(out, train_label)

            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Linear schedule with warmup
            if scheduler:
                scheduler.step()

            running_loss += train_loss

            train_accuracy += model_accuracy(train_label, out)
            train_f1_score += model_f1_score(train_label, out)

            # train_batch_count += 1
            # Saves train loss each 2 steps
            if (train_batch_count % 2) == 0:
                wandb.log({"train_loss": train_loss})

        # Validation step
        else:
            val_loss = 0
            val_accuracy = 0
            val_f1_score = 0
            with torch.no_grad():
                model.eval()
                for val_batch_count, val_sample in enumerate(tqdm(val_dataloader)):
                    val_audio, val_label = val_sample['image'].to(device), val_sample['target'].to(device)

                    out = model(val_audio)
                    out = out[embedding_layer]

                    loss = criterion(out, val_label)

                    val_accuracy += model_accuracy(val_label, out)
                    val_f1_score += model_f1_score(val_label, out)

                    val_loss += loss

                    # Saves val loss each 2 steps
                    if (val_batch_count % 2) == 0:
                        wandb.log({"val_loss": loss})

            # Log results on wandb
            wandb.log(
                {
                    "train_acc": (train_accuracy/len(train_dataloader))*100,
                    "val_acc": (val_accuracy/len(val_dataloader))*100,
                    "train_f1": train_f1_score/len(train_dataloader)*100,
                    "val_f1": val_f1_score/len(val_dataloader)*100,
                    "epoch": e
                }
            )

            log.info(
                'Train Accuracy: {:.3f} | Train F1-Score: {:.3f} | Train Loss: {:.6f} | Val Accuracy: {:.3f} | Val F1-Score: {:.3f} | Val loss: {:.6f}'.format(
                    (train_accuracy/len(train_dataloader))*100,
                    (train_f1_score/len(train_dataloader))*100,
                    running_loss/len(train_dataloader),
                    (val_accuracy/len(val_dataloader))*100,
                    (val_f1_score/len(val_dataloader))*100,
                    val_loss/len(val_dataloader)
                )
            )

            # Prints current learning rate value
            log.info(f"LR: {optimizer.param_groups[0]['lr']}")

            # Saves the model with the lowest val_loss value
            if val_loss/len(val_dataloader) < loss_min:
                log.info("Validation Loss Decreasead ({:.6f} --> {:.6f}), saving model...\n".format(loss_min, val_loss/len(val_dataloader)))
                loss_min = val_loss/len(val_dataloader)
                torch.save(
                    {
                        'epoch': epochs,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion
                    },
                    os.path.join(
                        weights_output_dir,
                        f'epochs_{epochs}-loss_{loss_min}-epoch_{e}.pth'
                    )
                )

            if early_stopping:
                current_patience += 1

                if current_patience == patience:
                    log.info("Early Stopping... ")
                    break

            model.train()