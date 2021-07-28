from question_answering.utils import get_dataset

def test_get_dataset() -> None:
    filename = "data/dev-v1.1.json"
    dataset = get_dataset(filename)
    assert 4930 == len(dataset)
