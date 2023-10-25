import math
import os
import random
import shutil
import string
from copy import deepcopy

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlInit)


def generate_random_string():
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(6))


def get_desired_dtype(dtype):
    if dtype == 'fp16':
        return torch.float16
    elif dtype == 'bf16':
        return torch.bfloat16
    else:
        return torch.float32


def print_line():
    prefix, unit, suffix = "#", "--", "#"
    print(prefix + unit*50 + suffix)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm%ds' % (m, s)


def execution_setup(cfg):
    print_line()
    if cfg.use_random_seed:
        seed = random.randint(401, 999)
        cfg.seed = seed

    print(f"setting seed: {cfg.seed}")
    seed_everything(cfg.seed)

    # if cfg.all_data:
    #     print("running training with all data...")
    #     fold = 0
    #     cfg.train_folds = [i for i in range(cfg.fold_metadata.n_folds)]
    #     cfg.valid_folds = [fold]
    #     cfg.outputs.model_dir = os.path.join(cfg.outputs.model_dir, f"all_data_training/seed_{cfg.seed}")
    # else:
    #     fold = cfg.fold
    #     cfg.train_folds = [i for i in range(cfg.fold_metadata.n_folds) if i != fold]
    #     cfg.valid_folds = [fold]
    # print(f"train folds: {cfg.train_folds}")
    # print(f"valid folds: {cfg.valid_folds}")

    # folder ---
    os.makedirs(cfg.outputs.model_dir, exist_ok=True)

    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def init_wandb(cfg):
    project = cfg.wandb.project
    tags = cfg.wandb.tags

    if cfg.wandb.all_data_flag:
        run_id = f"{cfg.wandb.run_name}-all-data"
    else:
        run_id = f"{cfg.wandb.run_name}"

    run = wandb.init(
        project=project,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=tags,
        name=run_id,
        anonymous="must",
        job_type="Train",
    )

    return run


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']*1e6


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(cfg, state, is_best):
    os.makedirs(cfg.outputs.model_dir, exist_ok=True)
    name = f"sci_model"

    filename = f'{cfg.outputs.model_dir}/{name}_last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)

    if is_best:
        shutil.copyfile(filename, f'{cfg.outputs.model_dir}/{name}_best.pth.tar')


class EMA():
    """
    credit: https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332567
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def sanitize_df(df):
    df = deepcopy(df)
    print(f"df shape before sanitize: {df.shape}")

    df['question_id'] = [f'q{idx}' for idx in range(len(df))]
    df = df[
        ~(df['prompt'].isna() | df['A'].isna() | df['B'].isna() | df['C'].isna() | df['D'].isna() | df['E'].isna())
    ].copy()

    df['valid_answer'] = df['answer'].apply(lambda x: x in ['A', 'B', 'C', 'D', 'E'])
    df = df[df['valid_answer']].copy()
    df = df.drop(columns=['valid_answer']).copy()
    df = df.reset_index(drop=True)

    keep_cols = ['question_id', 'prompt', 'A', 'B', 'C', 'D', 'E', 'answer']
    df = df[keep_cols].copy()

    df = df.reset_index(drop=True)
    print(f"df shape after sanitize: {df.shape}")

    return df


def sanitize_df_new(df):
    df = deepcopy(df)
    print(f"df shape before sanitize: {df.shape}")

    df = df[
        ~(df['prompt'].isna() | df['A'].isna() | df['B'].isna() | df['C'].isna() | df['D'].isna() | df['E'].isna())
    ].copy()

    df['valid_answer'] = df['answer'].apply(lambda x: x in ['A', 'B', 'C', 'D', 'E'])
    df = df[df['valid_answer']].copy()
    df = df.drop(columns=['valid_answer']).copy()
    df = df.reset_index(drop=True)

    df = df.reset_index(drop=True)
    print(f"df shape after sanitize: {df.shape}")

    return df


def shuffle_answer_key(df):
    shuffled_df = deepcopy(df)
    # print_line()
    print(f"Answer Key Distribution Before Shuffling: {shuffled_df.answer.value_counts().sort_index()}")

    key2idx = {v: k for k, v in enumerate(list("ABCDE"))}
    idx2key = {v: k for k, v in key2idx.items()}

    shuffled_df["answer_string"] = shuffled_df[["A", "B", "C", "D", "E", "answer"]].apply(
        lambda x: x[key2idx[x[-1]]], axis=1
    )

    shuffled_df["options"] = shuffled_df[["A", "B", "C", "D", "E"]].apply(
        lambda x: random.sample(list(x), len(x)), axis=1
    )

    shuffled_df["A"] = shuffled_df["options"].apply(lambda x: x[0])
    shuffled_df["B"] = shuffled_df["options"].apply(lambda x: x[1])
    shuffled_df["C"] = shuffled_df["options"].apply(lambda x: x[2])
    shuffled_df["D"] = shuffled_df["options"].apply(lambda x: x[3])
    shuffled_df["E"] = shuffled_df["options"].apply(lambda x: x[4])

    shuffled_df["answer"] = shuffled_df[["A", "B", "C", "D", "E", "answer_string"]].apply(
        lambda x: idx2key[[idx for idx in range(5) if x[idx] == x[-1]][0]], axis=1
    )

    shuffled_df = shuffled_df[df.columns].copy()
    shuffled_df = shuffled_df.reset_index(drop=True)

    print(f"Answer Key Distribution After Shuffling: {shuffled_df.answer.value_counts().sort_index()}")
    return shuffled_df
