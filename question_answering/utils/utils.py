from loguru import logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader



def train_epoch(model: nn.Module, iterator: DataLoader, criterion, optimizer, device, writer):
    model.train()
    epoch_loss = 0
    logger.info("train epoch started")
    for i, batch in enumerate(iterator):
        features, attention_mask, start_token, end_token = batch
        start_logits, end_logits = model(features.to(device), attention_mask.to(device))
        start_loss = criterion(start_logits, start_token.to(device))
        # start_loss.backward()
        end_loss = criterion(end_logits, end_token.to(device))
        # end_loss.backward()
        total_loss = (start_loss + end_loss) / 2
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss
        logger.debug("iteration {i} loss {l}", i=i, l=total_loss)
        writer.add_scalar('train loss', total_loss.item())
    return epoch_loss / len(iterator)


def validate(model: nn.Module, iterator: DataLoader, criterion, device, writer):
    model.eval()
    val_loss = 0
    logger.info("Eval started")
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            features, attention_mask, start_token, end_token = batch
            start_logits, end_logits = model(features.to(device), attention_mask.to(device))
            start_loss = criterion(start_logits, start_token.to(device))
            end_loss = criterion(end_logits, end_token.to(device))
            total_loss = (start_loss + end_loss) / 2
            val_loss += total_loss
            writer.add_scalar('validation loss', total_loss.item())
    return val_loss / len(iterator)

