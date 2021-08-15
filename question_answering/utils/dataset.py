import json
from typing import List, Mapping
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class TextClassificationDataset(Dataset):
    """
    Wrapper around Torch Dataset to perform text classification
    """
    def __init__(self,
                 texts: List[str],
                 start: List[int] = None,
                 end: List[int] = None,
                 label_dict: Mapping[str, int] = None,
                 max_seq_length: int = 512,
                 model_name: str = 'cointegrated/rubert-tiny') -> None:
        """
        Args:
            texts (List[str]): a list with texts to classify or to train the
                classifier on
            labels List[str]: a list with classification labels (optional)
            label_dict (dict): a dictionary mapping class names to class ids,
                to be passed to the validation data (optional)
            max_seq_length (int): maximal sequence length in tokens,
                texts will be stripped to this length
            model_name (str): transformer model name, needed to perform
                appropriate tokenization

        """

        self.texts = texts
        self.start = start
        self.end = end
        self.max_seq_length = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sep_vid = self.tokenizer.vocab["[SEP]"]
        self.cls_vid = self.tokenizer.vocab["[CLS]"]
        self.pad_vid = self.tokenizer.vocab["[PAD]"]

    def __len__(self):
        """
        Returns:
            int: length of the dataset
        """
        return len(self.texts)

    def __getitem__(self, index) -> List[torch.Tensor]: #Mapping[str, torch.Tensor]
        """Gets element of the dataset

        Args:
            index (int): index of the element in the dataset
        Returns:
            Single element by index
        """

        # encoding the text
        x = self.texts[index]
        x_encoded = self.tokenizer(
            x,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        input_ids = x_encoded["input_ids"].squeeze(0)
        true_seq_length = input_ids.size(0)
        pad_size = self.max_seq_length - true_seq_length
        pad_ids = torch.Tensor([self.pad_vid] * pad_size).long()
        x_tensor = torch.cat((input_ids, pad_ids))
        mask = torch.ones_like(input_ids, dtype=torch.int8)
        mask_pad = torch.zeros_like(pad_ids, dtype=torch.int8)
        mask = torch.cat((mask, mask_pad))
        start_token_idx = x_encoded.char_to_token(self.start[index])
        end_token_idx = x_encoded.char_to_token(self.end[index]) if x_encoded.char_to_token(self.end[index]) is not None else x_encoded.char_to_token(self.end[index] + 1)
        return [x_tensor, mask, start_token_idx, end_token_idx]


def get_dataset(filename: str,
                text_len_limit: int = 1500,
                max_seq_length: int = 512) -> TextClassificationDataset:


    with open(filename, "r") as fh:
        source = json.load(fh)
    raw_dict = {
        "text": [],
        "start": [],
        "end": []
    }
    for article in source["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                raw_dict["text"].append(para["context"] + "[SEP]" + qa["question"])
                raw_dict["start"].append(qa["answers"][0]["answer_start"])
                raw_dict["end"].append(qa["answers"][0]["answer_start"] + \
                                         len(qa["answers"][0]["text"]))
    df = pd.DataFrame(raw_dict)
    df["text_len"] = df["text"].apply(lambda x: len(x))
    df = df[df["text_len"] < text_len_limit]
    dataset = TextClassificationDataset(
        texts=df['text'].values.tolist(),
        start=df['start'].values.tolist(),
        end=df['end'].values.tolist(),
        max_seq_length=max_seq_length,
    )
    return dataset

