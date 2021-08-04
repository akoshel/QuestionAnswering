from question_answering.models import BertForQuestionAnswering

def test_berta_qa() -> None:
    model = BertForQuestionAnswering()
    assert model is not None
