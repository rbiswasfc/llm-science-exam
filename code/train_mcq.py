from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
import argparse
from datasets import Dataset
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer, EarlyStoppingCallback

import os
import random
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(42)

option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
index_to_option = {v: k for k,v in option_to_index.items()}

def preprocess(example):
    first_sentence = [f"{example['prompt']} {tokenizer.sep_token} {example[option]}" for option in 'ABCDE']
    second_sentences = [example['context']]*5
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation=True, max_length=MAX_LEN)
    tokenized_example['label'] = option_to_index[example['answer']]
    return tokenized_example

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch

def precision_at_k(r, k):
    assert k <= len(r)
    assert k != 0
    return sum(int(x) for x in r[:k]) / k

def compute_map3(eval_pred):
    """
    Score is mean average precision at 3
    """
    predictions, truths = eval_pred
    predictions = np.argsort(-predictions, 1)

    n_questions = len(predictions)
    score = 0.0

    for u in range(n_questions):
        user_preds = predictions[u]
        user_true = truths[u]

        user_results = [1 if item == user_true else 0 for item in user_preds]

        for k in range(min(len(user_preds), 3)):
            score += precision_at_k(user_results, k+1) * user_results[k]
    score /= n_questions
    score = round(score, 4)

    return {
        "map3": score,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_df", type=str, required=True)
    parser.add_argument("--valid_df", type=str, required=True)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--wd', type=float, default=0.1)
    parser.add_argument('--wr', type=float, default=0.1)
    parser.add_argument('--ep', type=int, default=3)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=1024)
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--freeze', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=26)
    parser.add_argument('--optim', type=str, default="adamw_torch")

    cargs = parser.parse_args()

    MODEL_NAME = cargs.model_name
    LR = cargs.lr
    WD = cargs.wd
    WR = cargs.wr
    EP = cargs.ep
    BS = cargs.bs
    OPTIM = cargs.optim
    FREEZE = cargs.freeze
    MAX_LEN = cargs.max_len
    PREFIX = cargs.prefix

    df_train = pd.read_parquet(cargs.train_df)
    df_valid = pd.read_parquet(cargs.valid_df)

    print("Training size:", len(df_train))
    print("Valid size:", len(df_valid))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if cargs.num_layers!=-1:
        model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, num_hidden_layers=cargs.num_layers, ignore_mismatched_sizes=True)
    else:
        model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True)

    train_dataset = Dataset.from_pandas(df_train)
    valid_dataset = Dataset.from_pandas(df_valid)

    train_dataset = train_dataset.map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
    valid_dataset = valid_dataset.map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])

    OUTPUT_DIR = f'checkpoint/{"_".join(MODEL_NAME.split("/"))}_lr_{LR}_wd_{WD}_wr_{WR}_ep_{EP}_maxlen_{MAX_LEN}_optim_{OPTIM}'
    if cargs.num_layers!=-1:
        OUTPUT_DIR += f"num_layers_{cargs.num_layers}"

    if (FREEZE > 0) and ("deberta" in MODEL_NAME):
        OUTPUT_DIR += f"_fr{FREEZE}"
        print("Freezing",FREEZE,"layers") 
        model.deberta.embeddings.requires_grad_(False)
        model.deberta.encoder.layer[:FREEZE].requires_grad_(False)
    elif (FREEZE > 0) and ("funnel" in MODEL_NAME):
        OUTPUT_DIR += f"_freeze"
        print("Freezing layers") 
        model.funnel.embeddings.requires_grad_(False)
        model.funnel.encoder.blocks[0][:5].requires_grad_(False)

    OUTPUT_DIR += f"{PREFIX}"

    training_args = TrainingArguments(
        warmup_ratio=WR,
        learning_rate=LR,
        weight_decay=WD,
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BS,
        per_device_eval_batch_size=BS*2,
        num_train_epochs=EP,
        fp16=True,
        metric_for_best_model="map3",
        greater_is_better=True,
        evaluation_strategy="steps",
        gradient_accumulation_steps=max(1, 16//BS),
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=100,
        report_to='tensorboard',
        save_total_limit=1,
        load_best_model_at_end=True,
        optim=OPTIM,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_map3,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()
