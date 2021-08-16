from torch.utils.data import DataLoader
from question_answering.utils import get_dataset
from question_answering.models import BertForQuestionAnswering


def test_get_dataset() -> None:
    filename = "data/dev-v1.1.json"
    dataset = get_dataset(filename)
    assert 4930 == len(dataset)
    model = BertForQuestionAnswering()
    train_dataloader = DataLoader(dataset=dataset,
                                  batch_size=2,
                                  shuffle=False,)
    iterator = iter(train_dataloader)
    # for i in range(len(train_dataloader)):
    #     print(i)
    #     if i == 22:
    #         print('here')
    #     _ = next(iterator)
    mini_batch = next(iter(train_dataloader))
    features, attention_mask, _, __ = mini_batch
    output = model(features, attention_mask)
    assert features.shape == output[0].shape == output[1].shape
