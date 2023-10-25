import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoConfig, AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention


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


class FeatureExtractor(nn.Module):
    """
    extract features from backbone outputs 
        - multi-head attention mechanism
        - weighted average of top transformer layers
    """

    def __init__(self, num_layers, hidden_size):
        super(FeatureExtractor, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # ------------ weighted-average ---------------------------------------------------#
        init_amax = 5
        weight_data = torch.linspace(-init_amax, init_amax, self.num_layers)
        weight_data = weight_data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.weights = nn.Parameter(weight_data, requires_grad=True)

        # ------------ multi-head attention -----------------------------------------------#
        attention_config = BertConfig()
        attention_config.update(
            {
                "num_attention_heads": 2,
                "hidden_size": self.hidden_size,
                "attention_probs_dropout_prob": 0.1,  # self.backbone.config.attention_probs_dropout_prob,
                "hidden_dropout_prob": 0.1,  # self.backbone.config.hidden_dropout_prob,
                "is_decoder": False,
            }
        )

        self.mha_layer_norm = nn.LayerNorm(self.hidden_size, 1e-7)
        self.attention = BertAttention(attention_config, position_embedding_type="absolute")

        # ------------ mean-pooling ------------------------------------------------------#
        self.pool = MeanPooling()

        # ------------ Layer Normalization ------------------------------------------------#
        self.layer_norm = nn.LayerNorm(self.hidden_size, 1e-7)

    def forward(self, backbone_outputs, attention_mask, span_head_idxs, span_tail_idxs):

        # ------------ Output Transformation ----------------------------------------------#
        x = torch.stack(backbone_outputs.hidden_states[-self.num_layers:])
        w = F.softmax(self.weights, dim=0)
        encoder_layer = (w * x).sum(dim=0)

        # ------------ Multi-head attention  ----------------------------------------------#
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoder_layer = self.mha_layer_norm(encoder_layer)
        encoder_layer = self.attention(encoder_layer, extended_attention_mask)[0]

        # ------------ mean-pooling  ------------------------------------------------------#

        feature_vector = []
        bs = encoder_layer.shape[0]

        for i in range(bs):
            span_vec_i = []
            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                tmp = torch.mean(encoder_layer[i, head:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_spans, h)
            feature_vector.append(span_vec_i)

        feature_vector = torch.stack(feature_vector)  # (bs, num_spans, h)

        return feature_vector


def reinit_deberta(backbone, num_reinit_layers):
    """
    reinitialize top `num_reinit_layers` of the backbone
    """
    config = backbone.config

    for layer in backbone.encoder.layer[-num_reinit_layers:]:
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()


class SciModel(nn.Module):
    """
    The feedback-ells model with separate task specific heads
    """

    def __init__(self, cfg):
        print("initializing the feedback model...")

        super(SciModel, self).__init__()
        self.cfg = cfg

        # ----------------------------- Backbone -----------------------------------------#
        backbone_config = AutoConfig.from_pretrained(self.cfg.model.backbone_path)
        self.backbone = AutoModel.from_pretrained(
            self.cfg.model.backbone_path,
            config=backbone_config
        )

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {cfg.model.len_tokenizer}")
        self.backbone.resize_token_embeddings(cfg.model.len_tokenizer)

        # enable gradient checkpointing
        self.backbone.gradient_checkpointing_enable()

        # re-initialization
        if cfg.model.num_layers_reinit > 0:
            print(f"re-initializing last {cfg.model.num_layers_reinit} layers of the base model...")
            reinit_deberta(self.backbone, cfg.model.num_layers_reinit)

        # freeze embeddings
        if cfg.model.n_freeze > 0:
            print(f"setting requires grad to false for first {cfg.model.n_freeze} layers")
            self.backbone.embeddings.requires_grad_(False)
            self.backbone.encoder.layer[:cfg.model.n_freeze].requires_grad_(False)

        # ----------------------------- Feature Extractor ---------------------------------#
        hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.01)

        # ----------------------------- Loss --------------------------------------------#
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.01)
        self.aux_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        span_head_idxs,
        span_tail_idxs,
        labels=None,
        aux_labels=None,
        **kwargs
    ):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        encoder_layer = outputs.last_hidden_state  # (bs, seq_len, hidden_size)

        feature_vector = []
        bs = encoder_layer.shape[0]

        for i in range(bs):
            span_vec_i = []
            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                tmp = torch.mean(encoder_layer[i, head:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_spans, h)
            feature_vector.append(span_vec_i)

        feature_vector = torch.stack(feature_vector)  # (bs, num_spans, h)
        feature_vector = self.dropout(feature_vector)

        # pdb.set_trace()
        logits = self.classifier(feature_vector)  # (bs, num_options, 1)
        logits = logits.squeeze(-1)  # (bs, num_options)
        # num_choices = logits.shape[1]
        # logits = logits.view(-1, num_choices)

        # loss
        loss_dict = dict()
        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            # loss = self.aux_loss_fn(logits, aux_labels)
            loss_dict["loss"] = loss

        return logits, loss, loss_dict


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# AWP
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class AWP:
    """Implements weighted adverserial perturbation
    adapted from: https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1, adv_eps=0.0001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, batch, accelerator):
        if self.adv_lr == 0:
            return
        self._save()
        self._attack_step()

        _, adv_loss, _ = self.model(**batch)
        self.optimizer.zero_grad()
        accelerator.backward(adv_loss)
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
