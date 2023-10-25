import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoConfig, AutoModel


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Rank Model
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class RankerModel(nn.Module):
    """
    The Sci-LLM re-ranker
    """

    def __init__(self, cfg):
        print("initializing the Rank Model...")

        super(RankerModel, self).__init__()
        self.cfg = cfg

        # ----------------------------- Backbone -----------------------------------------#
        backbone_config = AutoConfig.from_pretrained(self.cfg.model.backbone_path)
        backbone_config.update(
            {
                "use_cache": False,
            }
        )

        self.backbone = AutoModel.from_pretrained(self.cfg.model.backbone_path, config=backbone_config)
        self.backbone.gradient_checkpointing_enable()

        # Mean pooling
        self.pool = MeanPooling()

        # classifier
        num_features = self.backbone.config.hidden_size
        self.classifier = nn.Linear(num_features, 1)

        # loss function
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def encode(
        self,
        input_ids,
        attention_mask,
    ):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        encoder_layer = outputs.last_hidden_state
        embeddings = self.pool(encoder_layer, attention_mask)  # mean pooling

        return embeddings

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # features
        features = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # (bs, num_features)

        # logits
        logits = self.classifier(features).reshape(-1)
        loss_dict = dict()

        # loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            loss_dict = {"loss": loss}

        return logits, loss, loss_dict
