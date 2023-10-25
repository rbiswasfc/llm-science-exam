import gc
import json
import os
import pdb
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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

try:
    from e_topic.ret_dataset import (ContentDataset, QueryDataset,
                                     get_retriever_dataset)
    from e_topic.ret_loader import (ContentDataCollator, QueryDataCollator,
                                    RetrieverDataCollator, show_batch)
    from e_topic.ret_model import AWP, RetrieverModel
    from e_topic.ret_optimizer import get_optimizer
    from utils.metric_utils import compute_retrieval_metrics
    from utils.retrieve_utils import semantic_search
    from utils.train_utils import (AverageMeter, as_minutes, get_lr,
                                   init_wandb, print_gpu_utilization,
                                   print_line, save_checkpoint,
                                   seed_everything)

except Exception as e:
    print(e)
    raise ImportError

pd.options.display.max_colwidth = 1000

# -------- Evaluation -------------------------------------------------------------#


def run_evaluation(
    cfg,
    model,
    query_train_dl,
    query_dl,
    content_dl,
    label_df,
):

    label_df = deepcopy(label_df)
    gdf = label_df.groupby("query_id")["content_id"].apply(set).reset_index()
    gdf["content_id"] = gdf["content_id"].apply(lambda x: list(x))
    query2content = dict(zip(gdf["query_id"], gdf["content_id"]))

    # get validation query embeddings ---
    model.eval()

    query_ids = []
    query_embeddings = []

    progress_bar = tqdm(range(len(query_dl)))
    for batch in query_dl:
        with torch.no_grad():
            batch_query_ids = batch["query_id"]
            batch_query_embeddings = model.query_encoder.encode(batch, prefix="q")

        query_ids.append(batch_query_ids)
        query_embeddings.append(batch_query_embeddings)
        progress_bar.update(1)
    progress_bar.close()

    query_ids = list(chain(*query_ids))
    query_embeddings = torch.cat(query_embeddings, dim=0)
    print(f"shape of query embeddings: {query_embeddings.shape}")

    # ----
    query_ids_train = []
    query_embeddings_train = []

    progress_bar = tqdm(range(len(query_train_dl)))
    for batch in query_train_dl:
        with torch.no_grad():
            batch_query_ids = batch["query_id"]
            batch_query_embeddings = model.query_encoder.encode(batch, prefix="q")

        query_ids_train.append(batch_query_ids)
        query_embeddings_train.append(batch_query_embeddings)
        progress_bar.update(1)
    progress_bar.close()

    query_ids_train = list(chain(*query_ids_train))
    query_embeddings_train = torch.cat(query_embeddings_train, dim=0)
    print(f"shape of query embeddings (train): {query_embeddings_train.shape}")

    # get content embeddings ---
    content_ids = []
    progress_bar = tqdm(range(len(content_dl)))
    h_dim = query_embeddings.size(1)
    pooled_outputs = torch.empty([0, h_dim]).cuda()  # TODO: remove hardcoding

    for batch in content_dl:
        with torch.no_grad():
            batch_content_ids = batch["content_id"]
            batch_content_embeddings = model.content_encoder.encode(batch, prefix="c")
        content_ids.append(batch_content_ids)
        pooled_outputs = torch.cat([pooled_outputs, batch_content_embeddings], 0)

        torch.cuda.empty_cache()
        progress_bar.update(1)
    progress_bar.close()

    content_ids = list(chain(*content_ids))
    content_embeddings = pooled_outputs
    print(f"shape of content embeddings: {content_embeddings.shape}")

    # ------ evaluation ----------------------------------------------------------------#
    # top-k search ----------------------------------------------------------------------#
    results = semantic_search(
        query_embeddings,
        content_embeddings,
        query_chunk_size=64,
        corpus_chunk_size=200_000,
        top_k=cfg.model.n_neighbour,
    )
    # pdb.set_trace()

    true_content_ids = []
    pred_content_ids = []
    pred_scores = []

    for idx, re_i in enumerate(results):  # loop over query
        query_id = query_ids[idx]
        hit_i = [node['corpus_id'] for node in re_i]
        top_scores_i = [node['score'] for node in re_i]
        top_content_ids_i = [content_ids[pos] for pos in hit_i]
        pred_content_ids.append(top_content_ids_i)
        pred_scores.append(top_scores_i)
        true_content_ids.append(query2content[query_id])

    result_df = pd.DataFrame()
    result_df["query_id"] = query_ids
    result_df["true_ids"] = true_content_ids
    result_df["pred_ids"] = pred_content_ids
    result_df["pred_scores"] = pred_scores

    # compute metric ----
    eval_dict = dict()
    cutoffs = [1, 2, 5]

    for cutoff in cutoffs:
        cdf = result_df.copy()
        cdf["pred_ids"] = cdf["pred_ids"].apply(lambda x: x[:cutoff])
        m = compute_retrieval_metrics(cdf["true_ids"].values, cdf["pred_ids"].values)

        eval_dict[f"precision@{cutoff}"] = m["precision_score"]
        eval_dict[f"recall@{cutoff}"] = m["recall_score"]
        eval_dict[f"f1@{cutoff}"] = m["f1_score"]
        eval_dict[f"f2@{cutoff}"] = m["f2_score"]

    # get oof df
    oof_df = result_df.copy()
    oof_df = oof_df.drop(columns=["true_ids"])
    oof_df = oof_df.rename(columns={"pred_ids": "content_ids"})
    oof_df["content_ids"] = oof_df["content_ids"].apply(lambda x: " ".join(x))

    # hard negatives computations ----
    hard_negatives_map = dict()
    print(f"current negative depth: {cfg.model.negative_depth}")
    results = semantic_search(
        query_embeddings_train,
        content_embeddings,
        query_chunk_size=64,
        corpus_chunk_size=200_000,
        top_k=cfg.model.negative_depth,
    )

    for idx, re_i in enumerate(results):  # loop over query
        query_id = query_ids_train[idx]
        hit_i = [node['corpus_id'] for node in re_i]
        top_content_ids_i = [content_ids[pos] for pos in hit_i]

        true_ids = query2content[query_id]
        negative_ids = [x for x in top_content_ids_i if x not in true_ids]
        hard_negatives_map[query_id] = negative_ids

    with open("./hard_negatives_map.json", "w") as f:
        json.dump(hard_negatives_map, f)

    # ---------

    to_return = {
        "scores": eval_dict,
        "result_df": result_df,
        "oof_df": oof_df,
        "hard_negatives_map": hard_negatives_map,
    }

    return to_return


# -------- Main Function ---------------------------------------------------------#


@hydra.main(version_base=None, config_path="../conf/e_topic", config_name="conf_e_topic")
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
    query_df = pd.read_parquet(os.path.join(data_dir, "sci_queries.parquet"))
    content_df = pd.read_parquet(os.path.join(data_dir, "sci_contents.parquet"))
    labels_df = pd.read_parquet(os.path.join(data_dir, "sci_labels.parquet"))
    fold_df = pd.read_parquet(os.path.join(data_dir, "sci_folds.parquet"))

    columns = ["prompt", "A", "B", "C", "D", "E"]
    for col in columns:
        query_df[col] = query_df[col].fillna("")
    query_df['topic'] = query_df['topic'].astype(str)

    # ------- Data Split ----------------------------------------------------------------#
    print(f"shape of query before merge: {query_df.shape}")
    query_df = pd.merge(query_df, fold_df, on="query_id", how="left")
    print(f"shape of query after merge: {query_df.shape}")

    query_df_train = query_df[query_df["is_train"] == 1].copy()
    query_df_valid = query_df[query_df["is_train"] == 0].copy()

    query_df_train = query_df_train.reset_index(drop=True)
    query_df_valid = query_df_valid.reset_index(drop=True)

    print(f"shape of queries train data: {query_df_train.shape}")
    print(f"shape of queries validation data: {query_df_valid.shape}")

    train_query_ids = query_df_train["query_id"].unique().tolist()
    labels_df_train = labels_df[labels_df["query_id"].isin(train_query_ids)].copy()
    labels_df_valid = labels_df[~labels_df["query_id"].isin(train_query_ids)].copy()

    labels_df_train = labels_df_train.reset_index(drop=True)
    labels_df_valid = labels_df_valid.reset_index(drop=True)

    print(f"# of contents: {content_df['content_id'].nunique()}")
    train_content_ids = labels_df_train["content_id"].unique().tolist()
    print(f"# of train contents: {len(train_content_ids)}")

    gdf = labels_df_train.groupby("content_id")["query_id"].apply(list).reset_index()
    content2query = dict(zip(gdf["content_id"], gdf["query_id"]))
    gdf = labels_df_train.groupby("query_id")["content_id"].apply(list).reset_index()
    query2content = dict(zip(gdf["query_id"], gdf["content_id"]))

    gdf = query_df_train.groupby("topic")["query_id"].apply(set).reset_index()
    gdf["query_id"] = gdf["query_id"].apply(lambda x: list(x))
    topic2query = dict(zip(gdf["topic"], gdf["query_id"]))
    train_topics = list(topic2query.keys())

    # ------- Datasets ------------------------------------------------------------------#
    # The datasets for Dual Encoder
    # -----------------------------------------------------------------------------------#

    # 1. query dataset ------------------------------------------------------------------#
    query_ds_creator = QueryDataset(cfg)

    query_tokenizer = query_ds_creator.tokenizer
    cfg.model.query_encoder.len_tokenizer = len(query_tokenizer)

    query_train_ds = query_ds_creator.get_dataset(query_df_train)
    query_valid_ds = query_ds_creator.get_dataset(query_df_valid)

    # 2. content dataset ---------------------------------------------------------------#
    content_ds_creator = ContentDataset(cfg)
    content_tokenizer = content_ds_creator.tokenizer
    cfg.model.content_encoder.len_tokenizer = len(content_tokenizer)
    content_ds = content_ds_creator.get_dataset(content_df)

    # 3. Retrieval dataset --------------------------------------------------------------#
    retrieval_ds = get_retriever_dataset(cfg, labels_df_train)

    # ------- data collators ------------------------------------------------------------#
    query_collator = QueryDataCollator(tokenizer=query_tokenizer)
    content_collator = ContentDataCollator(tokenizer=content_tokenizer)

    try:
        with open("./hard_negatives_map.json", "r") as f:
            hard_negatives_map = json.load(f)

    except Exception as e:
        print(e)
        hard_negatives_map = dict()

    if len(hard_negatives_map) == 0:
        shift_start = 1
    else:
        shift_start = 0

    kwargs = dict(
        query_ds=query_train_ds,
        content_ds=content_ds,
        cfg=cfg,
        seed=cfg.seed,
        query2content=query2content,  # mapping build only using train set
        content2query=content2query,  # mapping build only using train set
        topic2query=topic2query,
        train_topics=train_topics,
        hard_negatives_map=hard_negatives_map,
        train_content_ids=train_content_ids,
    )

    retriever_collator = RetrieverDataCollator(
        tokenizer=query_tokenizer,
        kwargs=kwargs,
    )

    print_line()

    # ------- data loaders ----------------------------------------------------------------#
    # 1. valid query dataloader

    query_train_ds = query_train_ds.sort("input_length")
    query_train_ds.set_format(
        type=None,
        columns=[
            'query_id',
            'q_input_ids',
            'q_attention_mask',
        ]
    )

    query_train_dl = DataLoader(
        query_train_ds,
        batch_size=cfg.train_params.query_bs,
        shuffle=False,
        collate_fn=query_collator,
    )

    query_valid_ds = query_valid_ds.sort("input_length")
    query_valid_ds.set_format(
        type=None,
        columns=[
            'query_id',
            'q_input_ids',
            'q_attention_mask',
        ]
    )

    query_valid_dl = DataLoader(
        query_valid_ds,
        batch_size=cfg.train_params.query_bs,
        shuffle=False,
        collate_fn=query_collator,
    )

    # 2. content dataloader
    content_ds = content_ds.sort("input_length")
    content_ds.set_format(
        type=None,
        columns=[
            'content_id',
            'c_input_ids',
            'c_attention_mask',
        ]
    )

    content_dl = DataLoader(
        content_ds,
        batch_size=cfg.train_params.content_bs,
        shuffle=False,
        collate_fn=content_collator,
    )

    # 3. Retriever dataloader
    retrieval_ds.set_format(
        type=None,
        columns=[
            'query_id',
            'content_id',
        ]
    )

    retrieval_dl = DataLoader(
        retrieval_ds,
        batch_size=cfg.train_params.retriever_bs,
        shuffle=True,
        collate_fn=retriever_collator,
        # num_workers=1,
    )

    # ------- Wandb --------------------------------------------------------------------#
    if cfg.use_wandb:
        print("initializing wandb run...")
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        init_wandb(cfg)

    # --- show batch -------------------------------------------------------------------#
    print_line()

    print("showing a batch...")
    for b in retrieval_dl:
        break
    show_batch(b, query_tokenizer, content_tokenizer)

    # ------- Config -------------------------------------------------------------------#
    print("config for the current run")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(cfg_dict, indent=4))

    # ------- Model --------------------------------------------------------------------#
    print_line()
    print("creating the Sci-LLM query and content models...")
    model = RetrieverModel(cfg_dict)
    print_line()

    if cfg.model.load_from_ckpt:
        print("loading model from previously trained checkpoint...")
        checkpoint = cfg.model.ckpt_path
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['state_dict'])
        print(f"At onset model performance on validation set = {ckpt['lb']}")
        del ckpt
        gc.collect()

    # ------- Optimizer ----------------------------------------------------------------#
    print_line()
    print("creating the optimizer...")
    optimizer = get_optimizer(model, cfg_dict)

    # ------- Scheduler -----------------------------------------------------------------#
    print_line()
    print("creating the scheduler...")

    num_epochs = cfg_dict["train_params"]["num_epochs"]
    grad_accumulation_steps = cfg_dict["train_params"]["grad_accumulation"]
    warmup_pct = cfg_dict["train_params"]["warmup_pct"]

    num_update_steps_per_epoch = len(retrieval_dl)//grad_accumulation_steps
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
    # ------- AWP ----------------------------------------------------------------------#

    AWP_FLAG = False
    if cfg.awp.use_awp:
        awp = AWP(model, optimizer, adv_lr=cfg.awp.adv_lr, adv_eps=cfg.awp.adv_eps)

    # ------- Accelerator --------------------------------------------------------------#
    print_line()
    print("accelerator setup...")
    accelerator = Accelerator(mixed_precision='bf16')  # cpu = True

    model, optimizer, query_train_dl, query_valid_dl, content_dl, retrieval_dl = accelerator.prepare(
        model, optimizer, query_train_dl, query_valid_dl, content_dl, retrieval_dl
    )

    print("model preparation done...")
    print(f"current GPU utilization...")
    print_gpu_utilization()
    print_line()

    # ------- training setup --------------------------------------------------------------#
    best_lb = -1.  # track recall@1000
    save_trigger = cfg_dict["train_params"]["save_trigger"]

    patience_tracker = 0
    current_iteration = 0

    # ------- training  --------------------------------------------------------------------#
    start_time = time.time()

    for epoch in range(num_epochs):

        if (cfg.awp.use_awp) & (epoch >= cfg.awp.awp_trigger_epoch):
            print("AWP is triggered...")
            AWP_FLAG = True

        # close and reset progress bar
        if epoch != 0:
            progress_bar.close()

        time.sleep(1)
        progress_bar = tqdm(range(num_update_steps_per_epoch))
        loss_meter = AverageMeter()
        nce_loss_meter = AverageMeter()

        print_line()
        print(f"current epoch: {epoch+1}/{num_epochs}")
        print_line()

        # Training ------
        model.train()
        for step, batch in enumerate(retrieval_dl):
            # time.sleep(1)

            if step == 0:
                show_batch(batch, query_tokenizer, content_tokenizer)
                # print(f"len of hard negatives map: {len(retriever_collator.hard_negatives_map)}")

            loss, loss_dict = model(batch)
            accelerator.backward(loss)

            if AWP_FLAG:
                awp.attack_backward(batch, accelerator)

            # take optimizer and scheduler steps
            if (step + 1) % grad_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                loss_meter.update(loss.item())
                nce_loss_meter.update(loss_dict["nce"].item())

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
                    wandb.log({"nce_loss": round(nce_loss_meter.avg, 5)}, step=current_iteration)

            # >--------------------------------------------------|
            # >-- evaluation ------------------------------------|
            # >--------------------------------------------------|

            # if (current_iteration-1) % cfg_dict["train_params"]["eval_frequency"] == 0:
            if (current_iteration-shift_start) % cfg_dict["train_params"]["eval_frequency"] == 0:

                print("\n")
                print("GPU Utilization before evaluation...")
                print_gpu_utilization()

                # set model in eval mode
                model.eval()
                eval_response = run_evaluation(
                    cfg,
                    model,
                    query_train_dl,
                    query_valid_dl,
                    content_dl,
                    labels_df,
                )

                # update negative depth ---
                cfg.model.negative_depth = max(cfg.model.negative_depth-2, 16)
                print(f"current negative depth: {cfg.model.negative_depth}")
                retriever_collator.cfg.train_params.neg_ratio = min(
                    retriever_collator.cfg.train_params.neg_ratio + 2, 15)
                print(f"current negative ratio: {retriever_collator.cfg.train_params.neg_ratio}")

                scores_dict = eval_response["scores"]
                result_df = eval_response["result_df"]
                oof_df = eval_response["oof_df"]
                hard_negatives_map = eval_response["hard_negatives_map"]
                print(f"len of hard negatives map: {len(hard_negatives_map)}")
                print_line()

                retriever_collator.hard_negatives_map = hard_negatives_map  # deepcopy(hard_negatives_map)
                time.sleep(5)

                nce_loss_meter = AverageMeter()  # reset nce loss meter

                lb = scores_dict["recall@1"]

                print_line()
                et = as_minutes(time.time()-start_time)
                print(
                    f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}"
                )
                print_line()
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
                    oof_df.to_csv(os.path.join(cfg_dict["outputs"]["model_dir"], f"oof_df_best.csv"), index=False)
                    result_df.to_csv(os.path.join(cfg_dict["outputs"]["model_dir"],
                                     f"result_df_best.csv"), index=False)
                else:
                    print(f">>> patience reached {patience_tracker}/{cfg_dict['train_params']['patience']}")
                    print(f">>> current best score: {round(best_lb, 4)}")

                oof_df.to_csv(os.path.join(cfg_dict["outputs"]["model_dir"], f"oof_df_last.csv"), index=False)
                result_df.to_csv(os.path.join(cfg_dict["outputs"]["model_dir"], f"result_df_last.csv"), index=False)

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

                if (cfg.awp.use_awp) & (best_lb >= cfg.awp.awp_trigger):
                    print("AWP is triggered...")
                    AWP_FLAG = True

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
