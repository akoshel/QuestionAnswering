from loguru import logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch(model: nn.Module, iterator: DataLoader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    logger.info("train epoch started")
    iterator = iter(iterator)
    for i in range(len(iterator)):
        try:
            batch = next(iterator)
        except Exception as e:
            logger.warning("batch is bad {i} {e}", i=i, e=e)
            continue
        start_logits, end_logits = model(batch["features"].to(device),
                                         batch["attention_mask"].to(device))
        start_loss = criterion(start_logits, batch["start_token"].to(device))
        # start_loss.backward()
        end_loss = criterion(end_logits, batch["end_token"].to(device))
        # end_loss.backward()
        total_loss = (start_loss + end_loss) / 2
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss
        logger.debug("iteration {i} loss {l}", i=i, l=total_loss)
    return epoch_loss / len(iterator)


def validate(model: nn.Module, iterator: DataLoader, criterion, device):
    model.eval()
    val_loss = 0
    logger.info("Eval started")
    with torch.no_grad():
        for i in range(len(iterator)):
            try:
                batch = next(iterator)
            except Exception as e:
                logger.warning("batch is bad {i} {e}", i=i, e=e)
                continue
            start_logits, end_logits = model(batch["features"].to(device),
                                             batch["attention_mask"].to(device))
            start_loss = criterion(start_logits, batch["start_token"].to(device))
            end_loss = criterion(end_logits, batch["end_token"].to(device))
            total_loss = (start_loss + end_loss) / 2
            val_loss += total_loss
    return val_loss / len(iterator)
