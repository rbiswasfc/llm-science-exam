from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers.tokenization_utils_base import (PaddingStrategy,
                                                  PreTrainedTokenizerBase)


@dataclass
class SciCollator:
    """
    Data collator for llm-science-exam task
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        buffer_dict = dict()
        buffer_keys = ["question_id", "answer"]

        for key in buffer_keys:
            if key in features[0].keys():
                value = [feature[key] for feature in features]
                buffer_dict[key] = value

        # labels ---
        labels = None
        if "labels" in features[0].keys():
            labels = [feature["labels"] for feature in features]
            aux_labels = [feature["aux_labels"] for feature in features]

        span_head_idxs = [feature["span_head_idxs"] for feature in features]
        span_tail_idxs = [feature["span_tail_idxs"] for feature in features]

        features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            } for feature in features
        ]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,  # "pt",
        )

        # restore buffer --
        for key, value in buffer_dict.items():
            batch[key] = value

        batch['span_head_idxs'] = span_head_idxs
        batch['span_tail_idxs'] = span_tail_idxs

        tensor_keys = [
            "input_ids",
            "attention_mask",
            "span_head_idxs",
            "span_tail_idxs",
        ]

        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
            batch["aux_labels"] = torch.tensor(aux_labels, dtype=torch.float32)

        return batch


# -----


def show_batch(batch, tokenizer, num_examples=8, task="train"):

    print('=='*40)
    num_examples = min(num_examples, len(batch['question_id']))
    print(f"Showing {num_examples} from a {task} batch...")

    for i in range(num_examples):
        question_id = batch['question_id'][i]
        print('#---' + f" Question: {question_id}" + '---' * 40 + '#')

        question_text = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=False)
        print("question_text: ", question_text)

        print("--"*20)
        num_spans = len(batch['span_head_idxs'][i])
        mcq_keys = ['A', 'B', 'C', 'D', 'E']

        for span_idx in range(num_spans):
            start, end = batch['span_head_idxs'][i][span_idx], batch['span_tail_idxs'][i][span_idx]
            span = tokenizer.decode(batch['input_ids'][i][start:end])
            print(f"\n[Option {mcq_keys[span_idx]}]: {span}")

        if "infer" not in task.lower():
            print("--"*20)
            label = batch['labels'][i]
            print(f"Correct Option: {mcq_keys[label.item()]}")
            print(f"Aux labels: {batch['aux_labels'][i]}")

        print('=='*40)
