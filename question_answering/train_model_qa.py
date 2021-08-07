from loguru import logger
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from question_answering.utils import get_dataset, train_epoch
from question_answering.models import BertForQuestionAnswering
from question_answering.entities import read_model_params

def train(config_path: str='configs/config.yaml') -> None:
    config = read_model_params(config_path)
    train_dataset = get_dataset(config.data_params.train_data)
    logger.info("train dataset loaded len={l}", l=len(train_dataset))
    test_dataset = get_dataset(config.data_params.test_data)
    logger.info("test dataset loaded len={l}", l=len(test_dataset))
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=2,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=2,
                                  shuffle=True)
    model = BertForQuestionAnswering()
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_vid)
    optimizer = Adam(model.parameters())
    model.train()
    logger.info("here")
    for e in range(config.num_epoch):
        logger.info("epoch {e} started", e=e)
        train_epoch(model, train_dataloader, criterion, optimizer)




if __name__ == "__main__":
    logger.info("Question answering training started")
    train()
