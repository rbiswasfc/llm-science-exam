from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
    HfArgumentParser,
    BitsAndBytesConfig,
    GenerationConfig,
)

import time
import random
from dataclasses import dataclass, field
from typing import Optional
import torch
from functools import partial
from datasets import Dataset


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
            "help": "Modules to apply LoRA to. comma separated. e.g. 'q,k,v,o,wi,wo,shared'"
        },
    )

    max_seq_length: int = field(
        default=512,
        metadata={"help": "Max sequence length"},
    )

    answer_delimiter: str = field(
        default="extra_ids",
        metadata={"help": "Delimiter to use for answer choices. Use extra_ids or ABCDE"},
    )

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

    prompt_template_file: str = field(
        default="template1.txt",
        metadata={"help": "Path to prompt template file"},
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

    for context, prompt, answer_text, wrong_answers in zip(
        contexts, prompts, answer_texts, wrong_answer_texts
    ):
        options = [answer_text] + wrong_answers
        random.shuffle(options)

        answer_delimiters = "ABCDE"
        correct_tokens.append(answer_delimiters[options.index(answer_text)])
        full_prompts.append(
            template.format(
                context=context,
                prompt=prompt,
                A=options[0],
                B=options[1],
                C=options[2],
                D=options[3],
                E=options[4],
                answer=answer_delimiters[options.index(answer_text)],
            )
        )

    inputs = tokenizer(
        full_prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        pad_to_multiple_of=16,
    )

    labels = inputs.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    # labels = -100

    # labels[:, :-1] = -100

    # correct_tokens_ids = tokenizer(
    #         correct_tokens, 
    #         padding=False, 
    #         truncation=True,
    #         add_special_tokens=False,
    #         return_tensors="pt",
    #     ).input_ids
    
    # mask = inputs.input_ids == correct_tokens_ids

    # indices = mask.nonzero(as_tuple=True)

    # result = torch.full((labels.shape[0], ), -100, dtype=torch.long)
    # for idx, row in zip(indices[1], indices[0]):
    #     result[row] = idx

    # # should be all false, same shape as inputs.input_ids
    # mask2 = inputs.input_ids < 0 

    # mask2[torch.arange(labels.shape[0]), result] = True

    # # ignore anything other than the prediction
    # labels[~mask2] = -100


    return {
        **inputs,
        "labels": labels,
    }


def preprocess_text(batch, num_contexts=1):

    contexts = []

    for c in batch["context"]:
        # s = c.split("\n||\n")
        # temp = " ".join(s[0].split("|")[2:]).strip()
        # if num_contexts == 2:
        #     temp += " || " + " ".join(s[1].split("|")[2:]).strip()

        contexts.append(c)

    return {
        "context": contexts
    }


def load_data(cfg):

    if cfg.train_file.split(".")[-1].startswith("p"):
        load_func = Dataset.from_parquet
    elif cfg.train_file.split(".")[-1].startswith("c"):
        load_func = Dataset.from_csv

    train_dataset = load_func(cfg.train_file)
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
    id_preds, labels = eval_preds
    
    # correct_ids = labels[labels!=-100].tolist()

    #mistral
    correct_ids = []
    for l in labels.tolist():
        idx = -1
        while l[idx] == -100:
            idx -= 1
        correct_ids.append(l[idx]) 

    unq_ids = set(correct_ids)

    # id_preds = ((preds[0][:, 0, :])*-1).argsort()

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


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    
    # adds special token at end, so we need to the output 2 from the end
    return (-1*logits[:, -2, :]).argsort()

def main():
    parser = HfArgumentParser((Config, TrainingArguments))
    cfg, training_args = parser.parse_args_into_dataclasses()

    assert (
        training_args.remove_unused_columns is False
    ), "remove_unused_columns must be False"

    training_args.seed = int(time.time()%5000)

    set_seed(training_args.seed)

    torch_dtype = torch.float32
    q_config = None
    if cfg.model_dtype in {"fp16", "float16"}:
        torch_dtype = torch.float16
    elif cfg.model_dtype in {"bf16", "bfloat16"}:
        torch_dtype = torch.bfloat16
    elif cfg.model_dtype in {"int4"}:
        torch_dtype = torch.bfloat16
        q_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
    
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{free_in_GB-2}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path, 
            torch_dtype=torch_dtype,
            load_in_8bit=True if cfg.model_dtype == "int8" else False,
            quantization_config=q_config,
            use_flash_attention_2=True,
            trust_remote_code=True,
            token="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",
            # pretraining_tp=1,
        )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

    # training_args.generation_config = GenerationConfig.from_pretrained(cfg.model_name_or_path)

    # training_args.generation_config.max_new_tokens = 1
    # training_args.generation_config.max_length = cfg.max_seq_length
    # training_args.generation_config.return_dict_in_generate = True
    # training_args.generation_config.output_scores = True

    model.config.cfg = cfg.__dict__
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if cfg.use_lora:
        from peft import (
            LoraConfig, 
            get_peft_model, 
            prepare_model_for_kbit_training,
        )

        print("using lora")

        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0.05,
            target_modules=cfg.lora_modules.split(","),
        )
        model = get_peft_model(model, lora_cfg)

        if cfg.model_dtype in {"int8", "int4"}:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        model.print_trainable_parameters()

    train_dataset, val_dataset = load_data(cfg) 

    train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(800))
    val_dataset = val_dataset.select(range(500))

    with open(cfg.prompt_template_file, "r") as f:
        template = f.read()

    data_collator = partial(
            collate,
            tokenizer=tokenizer,
            max_length=cfg.max_seq_length,
            template=template,
        )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
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
