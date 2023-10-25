import pdb
import random
from copy import deepcopy
from dataclasses import dataclass, field

import torch
from transformers import DataCollatorWithPadding


@dataclass
class RetrieverDataCollator(DataCollatorWithPadding):
    """
    data collector for Sci-LLM content matching task
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"
    kwargs: field(default_factory=dict) = None

    def __post_init__(self):
        [setattr(self, k, v) for k, v in self.kwargs.items()]

        # mappings
        query2idx = dict()
        query_ids = self.query_ds["query_id"]
        self.all_query_ids = deepcopy(query_ids)

        for idx in range(len(query_ids)):
            query_id = query_ids[idx]
            query2idx[query_id] = idx

        content2idx = dict()
        content_ids = self.content_ds["content_id"]
        self.all_content_ids = deepcopy(self.train_content_ids)

        for idx in range(len(content_ids)):
            content_id = content_ids[idx]
            content2idx[content_id] = idx

        self.query2idx = query2idx
        self.content2idx = content2idx

        self.rng = random.Random(self.seed)

    def process_features(self, query_ids, content_ids):

        updated_features = []

        for query_id, content_id in zip(query_ids, content_ids):
            example = dict()

            example["query_id"] = query_id
            example["content_id"] = content_id

            # get fields
            ex_query_info = self.query_ds[self.query2idx[example["query_id"]]]
            ex_content_info = self.content_ds[self.content2idx[example["content_id"]]]

            # use fields
            example["q_input_ids"] = ex_query_info["q_input_ids"]
            example["q_attention_mask"] = ex_query_info["q_attention_mask"]

            example["c_input_ids"] = ex_content_info["c_input_ids"]
            example["c_attention_mask"] = ex_content_info["c_attention_mask"]

            updated_features.append(example)
        return updated_features

    def __call__(self, features):

        query_ids = [feature["query_id"] for feature in features]
        content_ids = [feature["content_id"] for feature in features]
        bs = len(query_ids)

        if self.rng.random() < self.cfg.train_params.topic_activation_th:
            num_topics = self.rng.randint(
                self.cfg.train_params.min_topic_num,
                self.cfg.train_params.max_topic_num,
            )

            selected_topics = self.rng.sample(self.train_topics, k=num_topics)
            query_pool = []
            for topic in selected_topics:
                query_pool += self.topic2query[topic]
            query_pool = list(set(query_pool))
            n = min(bs, len(query_pool))
            selected_query_ids = self.rng.sample(query_pool, k=n)

            rem = bs - n
            if rem > 0:
                rem_pool = list(set(self.all_query_ids) - set(selected_query_ids))
                selected_query_ids += self.rng.sample(rem_pool, k=rem)

            query_ids = []
            content_ids = []
            visited = set()

            for qid in selected_query_ids:
                cid = self.rng.sample(self.query2content[qid], k=1)[0]
                if cid not in visited:
                    visited.add(cid)
                    query_ids.append(qid)
                    content_ids.append(cid)
                else:
                    continue
        else:
            cutoff = len(query_ids) // (1 + self.cfg.train_params.neg_ratio)
            seed_query_ids = query_ids[:cutoff]
            seed_content_ids = content_ids[:cutoff]

            query_ids = deepcopy(seed_query_ids)
            content_ids = deepcopy(seed_content_ids)

            bs = len(seed_query_ids) * (1 + self.cfg.train_params.neg_ratio)

            # add in the negative samples ---
            for query_id, _ in zip(seed_query_ids, seed_content_ids):
                neg_pool = self.hard_negatives_map.get(query_id, [])
                neg_pool = list(set(neg_pool).intersection(set(self.all_content_ids)))

                if len(neg_pool) == 0:
                    print(f"no hard negatives for query_id: {query_id}")
                    selected_cids = self.rng.sample(self.all_content_ids, k=self.cfg.train_params.neg_ratio)
                elif len(neg_pool) < self.cfg.train_params.neg_ratio:
                    selected_cids = neg_pool
                    n_random = self.cfg.train_params.neg_ratio - len(neg_pool)
                    selected_cids += self.rng.sample(self.all_content_ids, k=n_random)
                else:
                    selected_cids = self.rng.sample(neg_pool, k=self.cfg.train_params.neg_ratio)

                for cid in selected_cids:
                    qid = self.rng.sample(self.content2query[cid], k=1)[0]
                    if (qid not in query_ids) & (cid not in content_ids):
                        query_ids.append(qid)
                        content_ids.append(cid)

            rem = bs - len(query_ids)
            if rem > 0:
                rem_pool = list(set(self.all_content_ids).difference(set(content_ids)))
                selected_cids = self.rng.sample(rem_pool, k=rem)
                for cid in selected_cids:
                    qid = self.rng.sample(self.content2query[cid], k=1)[0]
                    query_ids.append(qid)
                    content_ids.append(cid)

        # ----
        features = self.process_features(query_ids, content_ids)

        buffer_dict = dict()
        buffer_keys = ["query_id", "content_id"]

        for key in buffer_keys:
            value = [feature[key] for feature in features]
            buffer_dict[key] = value

        # ----
        q_features = [
            {
                "input_ids": feature["q_input_ids"],
                "attention_mask": feature["q_attention_mask"],

            } for feature in features
        ]

        # --- content
        c_features = []
        for feature in features:
            c_feature = {
                "input_ids": feature["c_input_ids"],
                "attention_mask": feature["c_attention_mask"]
            }

            c_features.append(c_feature)

        # --------
        q_batch = self.tokenizer.pad(
            q_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        c_batch = self.tokenizer.pad(
            c_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )
        # --------

        batch = dict()

        batch["q_input_ids"] = q_batch["input_ids"]
        batch["q_attention_mask"] = q_batch["attention_mask"]

        batch["c_input_ids"] = c_batch["input_ids"]
        batch["c_attention_mask"] = c_batch["attention_mask"]

        for key in buffer_keys:
            batch[key] = buffer_dict[key]

        tensor_keys = [
            "q_input_ids",
            "q_attention_mask",

            "c_input_ids",
            "c_attention_mask",
        ]

        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        return batch


@dataclass
class QueryDataCollator(DataCollatorWithPadding):
    """
    data collector for seq classification
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):
        buffer_dict = dict()
        buffer_keys = ["query_id"]

        for key in buffer_keys:
            value = [feature[key] for feature in features]
            buffer_dict[key] = value

        q_features = [
            {
                "input_ids": feature["q_input_ids"],
                "attention_mask": feature["q_attention_mask"],
            } for feature in features
        ]

        q_batch = self.tokenizer.pad(
            q_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        batch = dict()

        batch["q_input_ids"] = q_batch["input_ids"]
        batch["q_attention_mask"] = q_batch["attention_mask"]

        # ------
        for key in buffer_keys:
            batch[key] = buffer_dict[key]

        tensor_keys = ["q_input_ids", "q_attention_mask"]

        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)
        return batch


@dataclass
class ContentDataCollator(DataCollatorWithPadding):
    """
    data collector for seq classification
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):
        buffer_dict = dict()
        buffer_keys = ["content_id"]

        for key in buffer_keys:
            value = [feature[key] for feature in features]
            buffer_dict[key] = value

        c_features = [
            {
                "input_ids": feature["c_input_ids"],
                "attention_mask": feature["c_attention_mask"],

            } for feature in features
        ]

        c_batch = self.tokenizer.pad(
            c_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        batch = dict()
        batch["c_input_ids"] = c_batch["input_ids"]
        batch["c_attention_mask"] = c_batch["attention_mask"]

        # ------

        for key in buffer_keys:
            batch[key] = buffer_dict[key]

        tensor_keys = ["c_input_ids", "c_attention_mask"]

        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)
        return batch


# ---
def show_batch(batch, query_tokenizer, content_tokenizer, n_examples=16):

    bs = batch['q_input_ids'].size(0)
    print(f"batch size: {bs}")

    print(f"shape of q_input_ids: {batch['q_input_ids'].shape}")
    print(f"shape of c_input_ids: {batch['c_input_ids'].shape}")

    print("\n\n")
    for idx in range(n_examples):

        print("##"*40)
        print(f"Query:\n{query_tokenizer.decode(batch['q_input_ids'][idx], skip_special_tokens=True)}")
        print("\n")
        print(f"Content:\n{content_tokenizer.decode(batch['c_input_ids'][idx], skip_special_tokens=True)}")
        print("##"*40)
