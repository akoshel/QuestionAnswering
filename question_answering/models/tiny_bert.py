import torch.nn as nn
from transformers import AutoModel, DistilBertModel

class BertForQuestionAnswering(nn.Module):

    def __init__(self, pretrained_path: str = "distilbert-base-uncased"): #"cointegrated/rubert-tiny"
        super().__init__()
        self.num_labels = 2
        self.bert = DistilBertModel.from_pretrained(pretrained_path)  # BertModel(configs, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(312, self.num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
    ):
        outputs = self.bert(input_ids, attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        return start_logits, end_logits
