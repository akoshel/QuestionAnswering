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
                                  shuffle=True)
    iterator = iter(train_dataloader)
    for _ in range(len(train_dataloader)):
        try:
            _ = next(iterator)
        except Exception as e:
            print(f"here {e}")
    mini_batch = next(iter(train_dataloader))
    features, attention_mask, _, __ = mini_batch
    output = model(mini_batch['features'], mini_batch['attention_mask'])
    assert mini_batch['features'].shape == output[0].shape == output[1].shape
