import json
import os
import random
import time
from copy import deepcopy
from itertools import chain

import hydra
import pandas as pd
import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

try:
    from e_ranker.rank_dataset import RankerDataset
    from e_ranker.rank_loader import RankerCollator, show_batch
    from e_ranker.rank_model import RankerModel
    from e_ranker.rank_optimizer import get_optimizer
    from utils.metric_utils import compute_retrieval_metrics
    from utils.train_utils import (AverageMeter, as_minutes, get_lr,
                                   init_wandb, print_gpu_utilization,
                                   print_line, save_checkpoint,
                                   seed_everything)

except Exception as e:
    print(e)
    raise ImportError

pd.options.display.max_colwidth = 1000

# -------- Evaluation -------------------------------------------------------------#


def run_evaluation(cfg, model, valid_dl, query2content):
    model.eval()

    query_ids = []
    content_ids = []
    all_scores = []

    progress_bar = tqdm(range(len(valid_dl)))
    for batch in valid_dl:
        with torch.no_grad():
            batch_query_ids = batch["query_id"]
            batch_content_ids = batch["content_id"]

            logits, _, _ = model(**batch)  # .reshape(-1)
            logits = logits.reshape(-1)
            preds = torch.sigmoid(logits)

        query_ids.append(batch_query_ids)
        content_ids.append(batch_content_ids)
        all_scores.append(preds)
        progress_bar.update(1)
    progress_bar.close()

    # pdb.set_trace()
    query_ids = list(chain(*query_ids))
    content_ids = list(chain(*content_ids))
    all_scores = torch.cat(all_scores, dim=0)
    print(f"shape of score: {all_scores.shape}")
    all_scores = all_scores.cpu().numpy().tolist()

    result_df = pd.DataFrame()
    result_df["query_id"] = query_ids
    result_df["content_id"] = content_ids
    result_df["scores"] = all_scores

    agg_df = result_df.groupby("query_id")["content_id"].agg(list).reset_index()
    score_agg_df = result_df.groupby("query_id")["scores"].agg(list).reset_index()
    agg_df = pd.merge(agg_df, score_agg_df, on="query_id", how="left")

    agg_df = agg_df.rename(columns={"content_id": "pred_ids"})
    agg_df["true_ids"] = agg_df["query_id"].map(query2content)
    # agg_df["true_ids"] = agg_df["true_ids"].apply(lambda x: [x])  # convert to list

    # compute AUC score
    agg_df["y"] = agg_df[["true_ids", "pred_ids"]].apply(
        lambda x: [1 if y in x[0] else 0 for y in x[1]], axis=1
    )

    truths = list(chain(*agg_df["y"].values))
    preds = list(chain(*agg_df["scores"].values))

    fpr, tpr, thresholds = metrics.roc_curve(truths, preds)
    ranker_auc = metrics.auc(fpr, tpr)

    # compute metric
    eval_dict = dict()
    eval_dict["auc"] = ranker_auc

    # ---
    # compute metric ----
    cutoffs = [1, 2, 5]

    for cutoff in cutoffs:
        cdf = agg_df.copy()
        cdf["pred_ids"] = cdf["pred_ids"].apply(lambda x: x[:cutoff])
        m = compute_retrieval_metrics(cdf["true_ids"].values, cdf["pred_ids"].values)

        eval_dict[f"precision@{cutoff}"] = m["precision_score"]
        eval_dict[f"recall@{cutoff}"] = m["recall_score"]
        eval_dict[f"f1@{cutoff}"] = m["f1_score"]
        eval_dict[f"f2@{cutoff}"] = m["f2_score"]
    # ---

    # n_points = 50
    # n_step = 0.02
    # thresholds = [(i)*n_step for i in range(n_points)]

    # def filter_ids(pred_ids, scores, threshold):
    #     to_return = []
    #     for pid, s in zip(pred_ids, scores):
    #         if s >= threshold:
    #             to_return.append(pid)
    #     return to_return

    # best_f2 = -1.
    # best_m = None

    # for cutoff in thresholds:
    #     cdf = agg_df.copy()
    #     cdf["pred_ids"] = cdf[["pred_ids", "scores"]].apply(
    #         lambda x: filter_ids(x[0], x[1], cutoff), axis=1
    #     )
    #     m = compute_retrieval_metrics(cdf["true_ids"].values, cdf["pred_ids"].values)
    #     f2 = m["f2_score"]
    #     if f2 > best_f2:
    #         best_f2 = f2
    #         best_m = m
    #         best_m["threshold"] = cutoff

    # eval_dict[f"precision"] = best_m["precision_score"]
    # eval_dict[f"recall"] = best_m["recall_score"]
    # eval_dict[f"f1"] = best_m["f1_score"]
    # eval_dict[f"f2"] = best_m["f2_score"]
    # eval_dict[f"threshold"] = best_m["threshold"]

    print(eval_dict)

    # get oof df
    oof_df = agg_df.copy()
    oof_df = oof_df.drop(columns=["true_ids"])
    # oof_df["pred_ids"] = oof_df[["pred_ids", "scores"]].apply(
    #     lambda x: filter_ids(x[0], x[1], eval_dict[f"threshold"]), axis=1
    # )

    oof_df = oof_df.rename(columns={"pred_ids": "content_ids"})
    oof_df["content_ids"] = oof_df["content_ids"].apply(lambda x: " ".join(x))

    oof_df = oof_df[["query_id", "content_ids"]].copy()

    to_return = {
        "scores": eval_dict,
        "result_df": agg_df,
        "oof_df": oof_df,
    }

    return to_return


# -------- Main Function ---------------------------------------------------------#


@hydra.main(version_base=None, config_path="../conf/e_ranker", config_name="conf_e_ranker")
def run_training(cfg):
    # ------- Runtime Configs -----------------------------------------------------#
    print_line()
    if cfg.use_random_seed:
        seed = random.randint(401, 999)
        cfg.seed = seed

    print(f"setting seed: {cfg.seed}")
    seed_everything(cfg.seed)

    # ------- folder management --------------------------------------------------#
    os.makedirs(cfg.outputs.model_dir, exist_ok=True)

    # ------- load data ----------------------------------------------------------#
    print_line()
    data_dir = cfg.dataset.data_dir

    # load query dataframe
    data_df = pd.read_parquet(os.path.join(data_dir, "train_ranker_dataset.parquet"))
    data_df = data_df.rename(columns={"id": "query_id"})

    true_df = data_df[data_df["label"] == 1].copy()
    true_df = true_df.drop_duplicates(subset=["query_id", "content_id"])
    true_df = true_df.reset_index(drop=True)
    print(f"shape of true_df: {true_df.shape}")
    true_df = true_df.groupby("query_id")["content_id"].agg(list).reset_index()
    query2content = dict(zip(true_df["query_id"], true_df["content_id"]))

    # ------- Data Split ----------------------------------------------------------------#

    train_df = data_df[data_df["is_train"] == 0].copy()  # TODO: fix data split
    valid_df = data_df[data_df["is_train"] == 1].copy()

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    print(f"shape of train data: {train_df.shape}")
    print(f"shape of validation data: {valid_df.shape}")

    # # sample negatives ---
    # train_query_ids = list(set(train_df["query_id"].unique()))
    # train_query_ids = random.sample(train_query_ids, k=cfg.train_params.n_train_queries)
    # train_df = train_df[train_df["query_id"].isin(train_query_ids)].copy()
    # train_df = train_df.reset_index(drop=True)
    # print(f"shape of train data after sampling queries: {train_df.shape}")

    # neg_df = train_df[train_df["label"] == 0].copy()
    # pos_df = train_df[train_df["label"] == 1].copy()
    # n_neg = len(pos_df) * cfg.train_params.negative_sample_ratio
    # neg_df = neg_df.sample(n=n_neg, random_state=cfg.seed).reset_index(drop=True)
    # train_df = pd.concat([pos_df, neg_df], axis=0)
    # train_df = train_df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    # print(f"shape of train data after sampling: {train_df.shape}")

    # ------- Datasets ------------------------------------------------------------------#
    # The datasets for ranking
    # -----------------------------------------------------------------------------------#

    dataset_creator = RankerDataset(cfg)

    train_ds = dataset_creator.get_dataset(train_df)
    valid_ds = dataset_creator.get_dataset(valid_df)
    tokenizer = dataset_creator.tokenizer

    # ------- data loaders ----------------------------------------------------------------#
    data_collector = RankerCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=64
    )

    train_ds.set_format(
        type=None,
        columns=[
            'query_id',
            'content_id',
            'input_ids',
            'attention_mask',
            'label'
        ]
    )

    # sort valid dataset for faster evaluation
    valid_ds = valid_ds.sort("input_length")

    valid_ds.set_format(
        type=None,
        columns=[
            'query_id',
            'content_id',
            'input_ids',
            'attention_mask',
            'label'
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

    # ------- Wandb --------------------------------------------------------------------#
    if cfg.use_wandb:
        print("initializing wandb run...")
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        init_wandb(cfg)

    # --- show batch -------------------------------------------------------------------#
    print_line()

    for b in train_dl:
        break
    show_batch(b, tokenizer, task='training')

    print_line()

    for b in valid_dl:
        break
    show_batch(b, tokenizer, task='training')

    print_line()

    # ------- Config -------------------------------------------------------------------#
    print("config for the current run")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(cfg_dict, indent=4))

    # ------- Model --------------------------------------------------------------------#
    print_line()
    print("creating the Sci-LLM Ranker models...")
    model = RankerModel(cfg)
    print_line()

    # ------- Optimizer ----------------------------------------------------------------#
    print_line()
    print("creating the optimizer...")
    optimizer = get_optimizer(model, cfg)

    # ------- Scheduler -----------------------------------------------------------------#
    print_line()
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

    # ------- Accelerator --------------------------------------------------------------#
    print_line()
    print("accelerator setup...")
    accelerator = Accelerator(mixed_precision='bf16')  # cpu = True

    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl,
    )

    print("model preparation done...")
    print(f"current GPU utilization...")
    print_gpu_utilization()
    print_line()

    # ------- training setup --------------------------------------------------------------#
    best_lb = -1.  # track recall@1000
    save_trigger = cfg.train_params.save_trigger

    patience_tracker = 0
    current_iteration = 0

    # ------- training  --------------------------------------------------------------------#
    start_time = time.time()

    for epoch in range(num_epochs):
        # close and reset progress bar
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch))
        loss_meter = AverageMeter()

        # Training ------
        model.train()
        for step, batch in enumerate(train_dl):
            logits, loss, loss_dict = model(**batch)
            accelerator.backward(loss)

            # take optimizer and scheduler steps
            if (step + 1) % grad_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                loss_meter.update(loss.item())

                progress_bar.set_description(
                    f"STEP: {step+1:5}/{num_update_steps_per_epoch:5}. "
                    f"T-STEP: {current_iteration+1:5}/{num_training_steps:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"Loss: {loss_meter.avg:.4f}. "
                )

                progress_bar.update(1)
                current_iteration += 1

                if cfg.use_wandb:
                    wandb.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)
                    wandb.log({"lr": get_lr(optimizer)}, step=current_iteration)

            # >--------------------------------------------------|
            # >-- evaluation ------------------------------------|
            # >--------------------------------------------------|

            if (current_iteration-1) % cfg.train_params.eval_frequency == 0:
                print("\n")
                print("GPU Utilization before evaluation...")
                print_gpu_utilization()

                # set model in eval mode
                model.eval()
                eval_response = run_evaluation(
                    cfg,
                    model,
                    valid_dl,
                    query2content,
                )

                scores_dict = eval_response["scores"]
                result_df = eval_response["result_df"]
                oof_df = eval_response["oof_df"]

                lb = scores_dict["auc"]

                print_line()
                et = as_minutes(time.time()-start_time)
                print(
                    f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}"
                )
                print_line()
                print(f">>> Current LB (AUC) = {round(lb, 4)}")
                print(f">>> Current Recall@1 = {round(scores_dict['recall@1'], 4)}")
                print(f">>> Current recall@2 = {round(scores_dict['recall@2'], 4)}")
                print(f">>> Current recall@5 = {round(scores_dict['recall@5'], 4)}")

                print(f">>> Current F1@1 = {round(scores_dict['f1@1'], 4)}")
                print(f">>> Current F1@2 = {round(scores_dict['f1@2'], 4)}")
                print(f">>> Current F1@5 = {round(scores_dict['f1@5'], 4)}")

                print_line()

                is_best = False
                if lb >= best_lb:
                    best_lb = lb
                    is_best = True
                    patience_tracker = 0

                    # -----
                    best_dict = dict()
                    for k, v in scores_dict.items():
                        best_dict[f"{k}_at_best"] = v
                else:
                    patience_tracker += 1

                if is_best:
                    oof_df.to_csv(os.path.join(cfg.outputs.model_dir, f"oof_df_best.csv"), index=False)
                    result_df.to_csv(os.path.join(cfg.outputs.model_dir, f"result_df_best.csv"), index=False)
                else:
                    print(f">>> patience reached {patience_tracker}/{cfg_dict['train_params']['patience']}")
                    print(f">>> current best score: {round(best_lb, 4)}")

                oof_df.to_csv(os.path.join(cfg.outputs.model_dir, f"oof_df_last.csv"), index=False)
                result_df.to_csv(os.path.join(cfg.outputs.model_dir, f"result_df_last.csv"), index=False)

                # saving -----
                accelerator.wait_for_everyone()
                model = accelerator.unwrap_model(model)
                model_state = {
                    'step': current_iteration,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'lb': lb,
                }

                if best_lb > save_trigger:
                    save_checkpoint(cfg, model_state, is_best=is_best)

                # logging ----
                if cfg.use_wandb:
                    wandb.log({"lb": lb}, step=current_iteration)
                    wandb.log({"best_lb": best_lb}, step=current_iteration)

                    # -- log scores dict
                    for k, v in scores_dict.items():
                        wandb.log({k: round(v, 4)}, step=current_iteration)

                    # --- log best scores dict
                    for k, v in best_dict.items():
                        wandb.log({k: round(v, 4)}, step=current_iteration)

                # -- post eval
                model.train()
                torch.cuda.empty_cache()
                print_line()

                # early stopping ----
                if patience_tracker >= cfg_dict['train_params']['patience']:
                    print("stopping early")
                    model.eval()
                    return


if __name__ == "__main__":
    run_training()
