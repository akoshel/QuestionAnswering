from loguru import logger
import torch.nn as nn
from torch.optim import Adam
from question_answering.utils import get_dataset, train_epoch
from question_answering.models import BertForQuestionAnswering
from question_answering.entities import ModelParams, read_model_params

def train(config_path: str='configs/config.yaml') -> None:
    config = read_model_params(config_path)
    train_dataset = get_dataset(config.data_params.train_data)
    logger.info("train dataset loaded len={l}", l=len(train_dataset))
    test_dataset = get_dataset(config.data_params.test_data)
    logger.info("test dataset loaded len={l}", l=len(test_dataset))
    model = BertForQuestionAnswering()
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_vid)
    optimizer = Adam(model.parameters())
    model.train()
    logger.info("here")



if __name__ == "__main__":
    logger.info("Question answering training started")
    train()
