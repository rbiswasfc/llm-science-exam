from copy import deepcopy

from datasets import Dataset
from transformers import AutoTokenizer


def get_tokenizer(cfg, encoder_type="query"):
    """load the tokenizer"""

    if encoder_type == "query":
        tokenizer_path = cfg.model.query_encoder.backbone_path
        print(f"loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    elif encoder_type == "content":
        tokenizer_path = cfg.model.content_encoder.backbone_path
        print(f"loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    else:
        raise RuntimeError

    print(f"tokenizer len: {len(tokenizer)}")
    test_string = "This is a test\n"
    tokenized_string = tokenizer.tokenize(test_string)
    print(f"tokenizer test: {tokenized_string}")
    return tokenizer

# -----------------------------------------------------------------------------#
# ---------------  Retriever Dataset ------------------------------------------#
# -----------------------------------------------------------------------------#


def get_retriever_dataset(cfg, corr_df):
    print(f"shape of correlations: {corr_df.shape}")
    corr_df = deepcopy(corr_df)
    task_dataset = Dataset.from_pandas(corr_df)
    return task_dataset


# -----------------------------------------------------------------------------#
# ---------------  Query Dataset ----------------------------------------------#
# -----------------------------------------------------------------------------#


class QueryDataset:
    """
    Dataset class for the query encoder
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = get_tokenizer(self.cfg, encoder_type="query")

    def pre_process(self, df):
        columns = ["prompt", "A", "B", "C", "D", "E"]
        df["query"] = df[columns].apply(lambda x: " | ".join(x), axis=1)
        return df

    def tokenize_function(self, examples):
        tz = self.tokenizer(
            examples["query"],
            padding=False,
            truncation=True,
            max_length=self.cfg.model.query_encoder.max_length,
            add_special_tokens=True,
            return_token_type_ids=False,
        )

        to_return = {
            "q_input_ids": tz["input_ids"],
            "q_attention_mask": tz["attention_mask"],
        }

        return to_return

    def compute_input_length(self, examples):
        return {"input_length": [len(x) for x in examples["q_input_ids"]]}

    def get_dataset(self, query_df):
        """
        main api for creating the Sci LLM Query dataset
        """
        query_df = deepcopy(query_df)
        query_df = self.pre_process(query_df)

        query_dataset = Dataset.from_pandas(query_df)
        query_dataset = query_dataset.map(self.tokenize_function, batched=True)
        query_dataset = query_dataset.map(self.compute_input_length, batched=True)

        try:
            query_dataset = query_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            pass

        return query_dataset


# -----------------------------------------------------------------------------#
# ---------------  Content Dataset --------------------------------------------#
# -----------------------------------------------------------------------------#


class ContentDataset:
    """Dataset class for Sci-LLM content
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = get_tokenizer(self.cfg, encoder_type="content")

    def pre_process(self, df):
        # columns = ["page_title", "section_title", "content"]
        # df["content"] = df[columns].apply(lambda x: " | ".join(x), axis=1)
        df["content"] = df['context']  # .apply(lambda x: " | ".join(x), axis=1)

        return df

    def tokenize_function(self, examples):
        tz = self.tokenizer(
            examples["content"],
            padding=False,
            truncation=True,
            max_length=self.cfg.model.content_encoder.max_length,
            add_special_tokens=True,
            return_token_type_ids=False,
        )
        to_return = {
            "c_input_ids": tz["input_ids"],
            "c_attention_mask": tz["attention_mask"],
        }
        return to_return

    def compute_input_length(self, examples):
        return {"input_length": [len(x) for x in examples["c_input_ids"]]}

    def get_dataset(self, content_df):
        """
        main api for creating the Sci-LLM Content dataset
        """
        content_df = deepcopy(content_df)
        content_df = self.pre_process(content_df)

        content_dataset = Dataset.from_pandas(content_df)
        content_dataset = content_dataset.map(self.tokenize_function, batched=True)
        content_dataset = content_dataset.map(self.compute_input_length, batched=True)

        try:
            content_dataset = content_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            pass

        return content_dataset
