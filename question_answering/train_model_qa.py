import os
from loguru import logger
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import get_dataset, train_epoch, validate
from models import BertForQuestionAnswering
from entities import read_model_params
from torch.utils.tensorboard import SummaryWriter


def train(config_path: str='configs/config.yaml') -> None:
    writer = SummaryWriter()
    config = read_model_params(config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device is {device}", device=device)
    train_dataset = get_dataset(config.data_params.train_data)
    logger.info("train dataset loaded len={l}", l=len(train_dataset))
    test_dataset = get_dataset(config.data_params.test_data)
    logger.info("test dataset loaded len={l}", l=len(test_dataset))
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config.train_params.batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=config.train_params.batch_size,
                                  shuffle=False)
    model = BertForQuestionAnswering()
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_vid)
    optimizer = Adam(model.parameters(), lr=config.train_params.learning_rate)
    best_val_score = float("Inf")
    model.to(device)
    for e in range(config.train_params.num_epoch):
        logger.info("epoch {e} started", e=e)
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device, writer)
        logger.info("Train loss: {loss}", loss=train_loss)
        # val_score = validate(model, test_dataloader, criterion, device, writer)
        # logger.info("Val loss: {loss}", loss=val_score)
        # if val_score < best_val_score:
        #     best_val_score = val_score
        #     with open(os.path.join(config.train_params.model_checkpoint_path, "qa_model.pth"), "wb") as fp:
        #         torch.save(model.state_dict(), fp)


if __name__ == "__main__":
    logger.info("Question answering training started")
    train()
