from loguru import logger
import torch.nn as nn


def train_epoch(model: nn.Module, iterator, criterion, optimizer):
    model.train()
    epoch_loss = 0
    logger.info("train epoch started")
    for i, batch in enumerate(iterator):
        start_logits, end_logits = model(batch["features"], batch["attention_mask"])
        start_loss = criterion(start_logits, batch["start_token"])
        # start_loss.backward()
        end_loss = criterion(end_logits, batch["end_token"])
        # end_loss.backward()
        total_loss = (start_loss + end_loss) / 2
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss
    return epoch_loss / len(iterator)
