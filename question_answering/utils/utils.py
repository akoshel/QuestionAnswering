from loguru import logger
import torch.nn as nn


def train_epoch(model: nn.Module, iterator, optimizer):
    model.train()
    epoch_loss = 0
    logger.info("train epoch started")
    for i, batch in enumerate(iterator):
        output = model(batch["features"], batch["attention_mask"])
    ...