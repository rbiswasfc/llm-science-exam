import gc
import json
import os
import pdb
import time
from copy import deepcopy
from itertools import chain

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.checkpoint
import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

try:
    from r_delta.sci_dataset import SciDataset
    from r_delta.sci_loader import SciCollator, show_batch
    from r_delta.sci_model import AWP, SciModel
    from r_delta.sci_optimizer import get_optimizer
    from utils.metric_utils import get_score
    from utils.train_utils import (EMA, AverageMeter, as_minutes,
                                   execution_setup, get_lr, init_wandb,
                                   print_gpu_utilization, print_line,
                                   sanitize_df_new, save_checkpoint,
                                   shuffle_answer_key)

except Exception as e:
    print(e)
    raise ImportError


# -------- evaluation -------------------------------------------------------------------#


def run_evaluation(model, valid_dl):
    model.eval()

    question_ids = []

    all_preds = []
    all_labels = []

    mcq_keys = ['A', 'B', 'C', 'D', 'E']

    progress_bar = tqdm(range(len(valid_dl)))

    for batch in valid_dl:
        with torch.no_grad():
            # pdb.set_trace()
            batch_question_ids = batch["question_id"]
            batch_labels = batch["labels"].to('cpu').detach().numpy().tolist()
            batch_logits, _, _ = model(**batch)  # (batch, num_choices)
            batch_preds = torch.argsort(
                batch_logits,
                dim=1,
                descending=True
            )[:, :3].to('cpu').detach().numpy().tolist()

        # pdb.set_trace()
        question_ids.extend(batch_question_ids)
        all_preds.extend(batch_preds)
        all_labels.extend(batch_labels)

        progress_bar.update(1)
    progress_bar.close()

    # OOF & pred ---
    all_preds = [[mcq_keys[pred] for pred in preds] for preds in all_preds]
    all_labels = [mcq_keys[label] for label in all_labels]

    preds_df = pd.DataFrame()  # all_preds)
    preds_df["question_id"] = question_ids
    preds_df["prediction"] = all_preds
    preds_df["answer"] = all_labels

    lb = get_score(all_preds, all_labels)

    scores_dict = dict()
    scores_dict["lb"] = lb

    oof_df = preds_df[['question_id', 'prediction']].copy()
    oof_df = oof_df.rename(columns={"question_id": "id"})
    oof_df = oof_df.reset_index(drop=True)

    return scores_dict, oof_df, preds_df


# -------- Main Function ----------------------------------------------------------------#
@hydra.main(version_base=None, config_path="../conf/r_delta", config_name="conf_r_delta")
def run_training(cfg):
    cfg = execution_setup(cfg)

    # ------- Wandb ---------------------------------------------------------------------#
    if cfg.use_wandb:
        print("initializing wandb run...")
        init_wandb(cfg)

    # ------- load data -----------------------------------------------------------------#
    print_line()
    train_df = pd.read_csv(cfg.train_dataset_path)
    print(f"shape of train data: {train_df.shape}")
    train_df = sanitize_df_new(train_df)
    train_df = shuffle_answer_key(train_df)

    with open(cfg.train_support_path, 'r') as f:
        support_dict = json.load(f)
    train_df['support'] = train_df['id'].map(support_dict)
    assert train_df['support'].isna().sum() == 0, "support is missing / invalid"

    valid_df = pd.read_csv(cfg.valid_dataset_path)
    valid_df = valid_df[['id', 'prompt', 'A', 'B', 'C', 'D', 'E', 'answer']].copy()
    valid_df['id'] = valid_df['id'].astype(str)

    with open(cfg.valid_support_path, 'r') as f:
        support_dict = json.load(f)
    valid_df['support'] = valid_df['id'].map(support_dict)
    assert valid_df['support'].isna().sum() == 0, "support is missing / invalid"

    print(f"shape of train data: {train_df.shape}")
    print(f"shape of valid data: {valid_df.shape}")

    train_df = train_df.rename(columns={"id": "question_id"})
    valid_df = valid_df.rename(columns={"id": "question_id"})

    print_line()

    # ------- dataset -------------------------------------------------------------------#
    dataset_creator = SciDataset(cfg)

    train_ds = dataset_creator.get_dataset(train_df, mode="train")
    valid_ds = dataset_creator.get_dataset(valid_df, mode="valid")
    tokenizer = dataset_creator.tokenizer
    cfg.model.len_tokenizer = len(tokenizer)

    # ------- data loaders --------------------------------------------------------------#
    data_collector = SciCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=cfg.model.pad_to_multiple_of,
    )

    train_ds.set_format(
        type=None,
        columns=[
            'question_id',
            'input_ids',
            'attention_mask',
            'span_head_idxs',
            'span_tail_idxs',
            'labels',
            'aux_labels',
        ]
    )

    valid_ds.set_format(
        type=None,
        columns=[
            'question_id',
            'input_ids',
            'attention_mask',
            'span_head_idxs',
            'span_tail_idxs',
            'labels',
            'aux_labels',
        ]
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train_params.train_bs,
        shuffle=True,
        collate_fn=data_collector,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.train_params.valid_bs,
        shuffle=False,
        collate_fn=data_collector,
    )

    print("data preparation done...")
    print_line()

    # --- show batch---------------------------------------------------------------------#
    print_line()
    print("sample batch from train data")
    for batch in train_dl:
        show_batch(batch, tokenizer, task="training")
        break

    for batch in valid_dl:
        show_batch(batch, tokenizer, task="validation")
        break

    # ------- Model ---------------------------------------------------------------------#

    print_line()
    print("creating the commonlit model...")
    model = SciModel(cfg)
    print_line()

    if cfg.model.load_from_ckpt:
        print("loading model from previously trained checkpoint...")
        checkpoint = cfg.model.ckpt_path
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['state_dict'])
        print(f"At onset model performance on validation set = {ckpt['lb']}")
        del ckpt
        gc.collect()

    # ------- Optimizer -----------------------------------------------------------------#
    print_line()
    print("creating the optimizer...")
    optimizer = get_optimizer(model, cfg)

    # ------- Scheduler -----------------------------------------------------------------#
    print_line()
    print("creating the scheduler...")

    num_epochs = cfg.train_params.num_epochs
    grad_accumulation_steps = cfg.train_params.grad_accumulation
    warmup_pct = cfg.train_params.warmup_pct

    num_update_steps_per_epoch = len(train_dl)//grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct*num_training_steps)

    print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    print(f"# training steps: {num_training_steps}")
    print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # ------- AWP -----------------------------------------------------------------------#
    AWP_FLAG = False
    if cfg.awp.use_awp:
        awp = AWP(model, optimizer, adv_lr=cfg.awp.adv_lr, adv_eps=cfg.awp.adv_eps)

    # ------- Accelerator ---------------------------------------------------------------#
    print_line()
    print("accelerator setup...")
    if cfg.train_params.use_mixed_precision:
        print("using mixed precision training")
        accelerator = Accelerator(mixed_precision='bf16')
    else:
        accelerator = Accelerator()  # cpu = True

    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )

    print("model preparation done...")
    print(f"current GPU utilization...")
    print_gpu_utilization()
    print_line()

    # ------- EMA -----------------------------------------------------------------------------#
    if cfg.train_params.use_ema:
        print_line()
        decay_rate = cfg.train_params.decay_rate
        ema = EMA(model, decay=decay_rate)
        ema.register()

        print(f"EMA will be used during evaluation with decay {round(decay_rate, 4)}...")
        print_line()

    # ------- training setup ------------------------------------------------------------#
    print_line()
    print(OmegaConf.to_container(cfg, resolve=True))
    print_line()

    start_time = time.time()

    save_trigger = cfg.train_params.save_trigger
    patience_tracker = 0
    current_iteration = 0

    best_score = 0.

    # ------- Training -----------------------------------------------------------------------#

    for epoch in range(num_epochs):
        epoch_progress = 0
        # close and reset progress bar
        if epoch != 0:
            progress_bar.close()

        # AWP Flag check
        if (cfg.awp.use_awp) & (epoch >= cfg.awp.awp_trigger_epoch):
            print("AWP is triggered...")
            AWP_FLAG = True

        progress_bar = tqdm(range(num_update_steps_per_epoch))

        loss_meter = AverageMeter()

        # Training
        model.train()

        for step, batch in enumerate(train_dl):
            logits, loss, loss_dict = model(**batch)
            accelerator.backward(loss)
            epoch_progress += 1

            if AWP_FLAG:
                awp.attack_backward(batch, accelerator)

            if (step + 1) % grad_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip_value)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                loss_meter.update(loss.item())

                if cfg.train_params.use_ema:
                    ema.update()

                progress_bar.set_description(
                    f"STEP: {epoch_progress+1:5}/{len(train_dl):5}. "
                    f"T-STEP: {current_iteration+1:5}/{num_training_steps:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"Loss: {loss_meter.avg:.4f}. "
                )

                progress_bar.update(1)
                current_iteration += 1

                if cfg.use_wandb:
                    wandb.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)

                    wandb.log({"lr": get_lr(optimizer)}, step=current_iteration)

            # Evaluation
            if (epoch_progress - 1) % cfg.train_params.eval_frequency == 0:
                print("\n")
                print("GPU Utilization before evaluation...")
                print_gpu_utilization()

                # set model in eval mode
                model.eval()

                # apply ema if it is used
                if cfg.train_params.use_ema:
                    ema.apply_shadow()

                scores_dict, oof_df, preds_df = run_evaluation(model, valid_dl)

                is_best = False
                if scores_dict['lb'] > best_score:
                    best_score = scores_dict['lb']
                    is_best = True
                    patience_tracker = 0
                else:
                    patience_tracker += 1

                print_line()
                et = as_minutes(time.time()-start_time)

                print(f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}")
                print(f">>> Current LB = {round(scores_dict['lb'], 4)}")

                if cfg.use_wandb:
                    wandb.log({"lb": scores_dict['lb']}, step=current_iteration)

                # save model
                accelerator.wait_for_everyone()
                model = accelerator.unwrap_model(model)
                model_state = {
                    'step': step + 1,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'lb': scores_dict['lb'],
                }

                if is_best:
                    oof_df.to_csv(os.path.join(cfg.outputs.model_dir, f"oof_df_best.csv"), index=False)
                    preds_df.to_csv(os.path.join(cfg.outputs.model_dir, f"preds_df_best.csv"), index=False)
                else:
                    print(f">>> patience reached {patience_tracker}/{cfg.train_params.patience}")
                    print(f">>> current best score: {round(best_score, 4)}")

                oof_df.to_csv(os.path.join(cfg.outputs.model_dir, f"oof_df_last.csv"), index=False)
                preds_df.to_csv(os.path.join(cfg.outputs.model_dir, f"preds_df_last.csv"), index=False)

                save_checkpoint(cfg, model_state, is_best=is_best)

                if cfg.use_wandb:
                    wandb.log({"best_lb": best_score}, step=current_iteration)

                if (cfg.awp.use_awp) & (best_score <= cfg.awp.awp_trigger):
                    print("AWP is triggered...")
                    AWP_FLAG = True

                torch.cuda.empty_cache()

                if cfg.train_params.use_ema:
                    ema.restore()

                model.train()
                print("GPU Utilization after evaluation...")
                print_gpu_utilization()
                print_line()

                if patience_tracker >= cfg.train_params.patience:
                    print("stopping early")
                    model.eval()
                    return


if __name__ == "__main__":
    run_training()
