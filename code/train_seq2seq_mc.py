from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
    HfArgumentParser,
    GenerationConfig,
    BitsAndBytesConfig
)

import time
import random
from dataclasses import dataclass, field
from typing import Optional
import torch
from functools import partial
from datasets import Dataset
import numpy as np
from accelerate import Accelerator

@dataclass
class Config:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )

    model_dtype: str = field(
        default="fp32",
        metadata={"help": "Model data type"},
    )

    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for training"},
    )

    lora_alpha: float = field(
        default=0.5,
        metadata={"help": "Alpha value for LoRA"},
    )

    lora_r: int = field(
        default=8,
        metadata={"help": "R value for LoRA"},
    )

    lora_modules: str = field(
        default="q,k,v",
        metadata={
            "help": "Modules to apply LoRA to. comma separated. e.g. 'q,k,v,o,wo,wi_0,wi_1,shared'"
        },
    )

    max_seq_length: int = field(
        default=512,
        metadata={"help": "Max sequence length"},
    )

    # answer_delimiter: str = field(
    #     default="extra_ids",
    #     metadata={"help": "Delimiter to use for answer choices. Use extra_ids or ABCDE"},
    # )

    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to training file"},
    )
    val_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to validation file"},
    )

    num_proc: int = field(
        default=4,
        metadata={"help": "Number of processes to use for preprocessing"},
    )

    num_contexts: int = field(
        default=1,
        metadata={"help": "Number of contexts to use"},
    )

    template_path: str = field(
        default=None,
        metadata={"help": "Path to template for encoder"},
    )


def collate(examples, tokenizer, max_length, template):
    """
    Have prompt, context, ABCDE.
    Want to shuffle ABCDE.
    """

    prompts = [e["prompt"] for e in examples]
    contexts = [e["context"] for e in examples]

    answer_texts = []
    wrong_answer_texts = []

    for e in examples:
        answer_letter = e["answer"]
        answer_texts.append(e[answer_letter])
        wrong_answers = [e[letter] for letter in "ABCDE" if letter != answer_letter]
        wrong_answer_texts.append(wrong_answers)

    full_prompts = []
    correct_tokens = []

    for prompt, answer_text, wrong_answers in zip(
        prompts, answer_texts, wrong_answer_texts
    ):
        options = [answer_text] + wrong_answers
        random.shuffle(options)

        # random idea to have <extra_id_> serve as delimters
        # if answer_delimiter == "extra_ids":
        #     answer_delimiters = [f"<extra_id_{n}>" for n in range(5)]
        # else:
        #     
        answer_delimiters = "ABCDE"
        
        correct_tokens.append(answer_delimiters[options.index(answer_text)])

        full_prompts.append(template.format(
            prompt=prompt,
            A=options[0],
            B=options[1],
            C=options[2],
            D=options[3],
            E=options[4],
        ))

    try:
        inputs = tokenizer(
            contexts,
            full_prompts,
            padding=True,
            truncation="only_first",
            max_length=max_length,
            return_tensors="pt",
            pad_to_multiple_of=16,
        )
    # there are some cases where the question and answers are really long.
    # in these cases, truncating just the context won't be enough.
    except:
        inputs = tokenizer(
            full_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            pad_to_multiple_of=16,
        )

    labels = tokenizer(
        correct_tokens,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        pad_to_multiple_of=16,
    )["input_ids"]

    labels[labels == tokenizer.pad_token_id] = -100

    return {
        **inputs,
        "labels": labels,
    }

def preprocess_text(batch, num_contexts=1):

    contexts = []

    for c in batch["context"]:
        s = c.split("\n||\n")
        temp = " ".join(s[0].split("|")[2:]).strip()
        if num_contexts == 2:
            temp += " || " + " ".join(s[1].split("|")[2:]).strip()

        contexts.append(temp)

    return {
        "context": contexts
    }


def load_data(cfg):
    train_dataset = Dataset.from_parquet(cfg.train_file)
    val_dataset = Dataset.from_parquet(cfg.val_file)

    train_dataset = train_dataset.map(
        preprocess_text, 
        num_proc=cfg.num_proc, 
        batched=True,
        fn_kwargs={"num_contexts": cfg.num_contexts},
        )
    val_dataset = val_dataset.map(
        preprocess_text, 
        num_proc=cfg.num_proc, 
        batched=True,
        fn_kwargs={"num_contexts": cfg.num_contexts},
        )

    return train_dataset, val_dataset


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    correct_ids = labels[:, 0]

    unq_ids = set(correct_ids.tolist())

    id_preds = ((preds[0][:, 0, :])*-1).argsort()

    final_preds = []
    for row in id_preds:
        temp = []
        for i in row:
            if i in unq_ids:
                temp.append(i)
            if len(temp) == 3: break
        final_preds.append(temp)

    score = 0
    for p, correct in zip(final_preds, correct_ids):
        if p[0] == correct:
            score += 1
        elif p[1] == correct:
            score += 0.5
        elif p[2] == correct:
            score += 0.33

    return {"map@3": round(score/len(final_preds), 5)}




def main():
    parser = HfArgumentParser((Config, Seq2SeqTrainingArguments))
    cfg, training_args = parser.parse_args_into_dataclasses()

    assert (
        training_args.remove_unused_columns is False
    ), "remove_unused_columns must be False"

    training_args.seed = int(time.time()%5000)


    set_seed(training_args.seed)

    q_config = None
    torch_dtype = torch.float32
    if cfg.model_dtype in {"fp16", "float16"}:
        torch_dtype = torch.float16
    elif cfg.model_dtype in {"bf16", "bfloat16"}:
        torch_dtype = torch.bfloat16
    elif cfg.model_dtype in {"int4"}:
        torch_dtype = torch.bfloat16
        q_config = BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

    model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg.model_name_or_path, 
            torch_dtype=torch_dtype,
            load_in_8bit=True if cfg.model_dtype == "int8" else False,
            quantization_config=q_config
            # device_map={"": Accelerator().process_index}
        )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

    model.config.cfg = cfg.__dict__
    model.config.decoder_start_token_id = 0

    if cfg.use_lora:
        from peft import (
            LoraConfig, 
            get_peft_model, 
            prepare_model_for_kbit_training,
        )

        print("using lora")

        lora_cfg = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            inference_mode=False,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0.1,
            target_modules=cfg.lora_modules.split(","),
        )
        model = get_peft_model(model, lora_cfg)

        if cfg.model_dtype in {"int8", "int4"}:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        model.print_trainable_parameters()

        # necessary for gradient checkpointing
        # https://github.com/huggingface/peft/issues/522#issuecomment-1705989330
        model.base_model.model.encoder.enable_input_require_grads()
        model.base_model.model.decoder.enable_input_require_grads()

    train_dataset, val_dataset = load_data(cfg) 

    train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(2000))

    with open(cfg.template_path) as fp:
        template = fp.read()

    data_collator = partial(
            collate,
            tokenizer=tokenizer,
            max_length=cfg.max_seq_length,
            template=template,
        )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    idx = random.randint(1, len(train_dataset)-1)

    tokenized = data_collator([train_dataset[idx-1], train_dataset[idx]])

    labels = tokenized["labels"]
    labels[labels == -100] = tokenizer.pad_token_id

    print(tokenizer.batch_decode(tokenized["input_ids"]))
    print(tokenizer.batch_decode(labels))


    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)



if __name__ == "__main__":
    main()
