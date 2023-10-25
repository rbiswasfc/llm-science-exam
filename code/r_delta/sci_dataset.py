import pdb
from copy import deepcopy

from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer


# --------------- Tokenizer ---------------------------------------------#
def get_tokenizer(cfg):
    """load the tokenizer"""
    tokenizer_path = cfg.model.backbone_path
    print(f"loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # NEW TOKENS
    new_tokens = [cfg.model.span_start_token, cfg.model.span_end_token]
    print("adding new tokens...")
    tokens_to_add = []
    for this_tok in new_tokens:
        tokens_to_add.append(AddedToken(this_tok, lstrip=False, rstrip=False))
    tokenizer.add_tokens(tokens_to_add)

    print(f"tokenizer len: {len(tokenizer)}")

    test_string = f"This is a test {' '.join(new_tokens)}"
    tokenized_string = tokenizer.tokenize(test_string)
    print(f"test string: {test_string}")
    print(f"tokenizer test: {tokenized_string}")
    return tokenizer

# --------------- Dataset ----------------------------------------------#


class SciDataset:
    """
    Dataset class for llm-science-exam task
    """

    def __init__(self, cfg):
        # assign config
        self.cfg = cfg

        self.choices = list('ABCDE')
        self.num_choices = len(self.choices)

        self.option_to_index = {option: idx for idx, option in enumerate(self.choices)}

        # tokenizer
        self.tokenizer = get_tokenizer(cfg)

        self.span_start_id = self.tokenizer.convert_tokens_to_ids(cfg.model.span_start_token)
        self.span_end_id = self.tokenizer.convert_tokens_to_ids(cfg.model.span_end_token)

    def preprocess_function(self, examples):
        input_columns = ['prompt', 'support', 'A', 'B', 'C', 'D', 'E']

        tokenized_inputs = dict()

        for col in input_columns:
            if col == 'prompt':
                max_length = self.cfg.model.max_length_prompt
            elif col == 'support':
                max_length = self.cfg.model.max_length_support
            else:
                max_length = self.cfg.model.max_length_option

            tz = self.tokenizer(
                examples[col],
                padding=False,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
                return_token_type_ids=False,
            )
            tokenized_inputs[col] = deepcopy(tz)
            # tokenized_inputs.append(deepcopy(tz))

        input_ids = []
        attention_mask = []
        span_head_idxs = []
        span_tail_idxs = []

        num_examples = len(examples['prompt'])

        for qid in range(num_examples):
            ex_input_ids = [self.tokenizer.cls_token_id]
            ex_attention_mask = [1]
            ex_span_head_idxs = []
            ex_span_tail_idxs = []

            ex_input_ids += tokenized_inputs['prompt']['input_ids'][qid]
            ex_attention_mask += tokenized_inputs['prompt']['attention_mask'][qid]

            ex_input_ids += [self.tokenizer.sep_token_id]
            ex_attention_mask += [1]

            ex_input_ids += tokenized_inputs['support']['input_ids'][qid]
            ex_attention_mask += tokenized_inputs['support']['attention_mask'][qid]

            for col in self.choices:
                ex_input_ids += [self.span_start_id]
                ex_attention_mask += [1]
                ex_span_head_idxs.append(len(ex_input_ids))

                ex_input_ids += tokenized_inputs[col]['input_ids'][qid]
                ex_attention_mask += tokenized_inputs[col]['attention_mask'][qid]

                ex_input_ids += [self.span_end_id]
                ex_attention_mask += [1]
                ex_span_tail_idxs.append(len(ex_input_ids)-1)

            ex_input_ids += [self.tokenizer.sep_token_id]
            ex_attention_mask += [1]

            input_ids.append(ex_input_ids)
            attention_mask.append(ex_attention_mask)
            span_head_idxs.append(ex_span_head_idxs)
            span_tail_idxs.append(ex_span_tail_idxs)
        # pdb.set_trace()

        # prepare return dict
        to_return = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "span_head_idxs": span_head_idxs,
            "span_tail_idxs": span_tail_idxs,
        }

        return to_return

    def generate_labels(self, examples):
        labels = [self.option_to_index[ans] for ans in examples['answer']]
        return {"labels": labels}

    def generate_aux_labels(self, examples):
        aux_labels = [[float(key == ans) for key in self.choices] for ans in examples['answer']]
        return {"aux_labels": aux_labels}

    def compute_input_length(self, examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def get_dataset(self, df, mode='train'):
        """
        Main api for creating the Science Exam dataset
        :param df: input dataframe
        :type df: pd.DataFrame
        :return: the created dataset
        :rtype: Dataset
        """
        df = deepcopy(df)
        task_dataset = Dataset.from_pandas(df)
        task_dataset = task_dataset.map(self.preprocess_function, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)

        # pdb.set_trace()

        if mode != "infer":
            task_dataset = task_dataset.map(self.generate_labels, batched=True)
            task_dataset = task_dataset.map(self.generate_aux_labels, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            print(e)

        return task_dataset
