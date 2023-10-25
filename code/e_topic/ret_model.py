
import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import AutoConfig, AutoModel


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Utils
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Pooling
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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
# Retriever Model
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class RetrieverModel(nn.Module):
    """
    The Sci-LLM dual-encoder model for retrieval task 
    """

    def __init__(self, config):
        print("initializing the Sci-LLM Retriever model...")

        super(RetrieverModel, self).__init__()
        self.config = config

        # --- Query Encoder ------------------------------------------------------------#
        self.query_encoder = SciEncoder(config["model"]["query_encoder"])

        # --- Content Encoder ----------------------------------------------------------#
        self.content_encoder = SciEncoder(config["model"]["content_encoder"])

        # --- Similarity ---------------------------------------------------------------#
        self.temperature = config["model"]["temperature"]
        self.similarity_fn = Similarity(temp=self.temperature)

        # --- Loss function ------------------------------------------------------------#
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        device = batch["q_input_ids"].get_device()

        q_embeddings = self.query_encoder.encode(batch, prefix="q")  # (b, h)
        c_embeddings = self.content_encoder.encode(batch, prefix="c")  # (b, h)

        logits1 = self.similarity_fn(
            x=q_embeddings.unsqueeze(1),  # (b, 1, h)
            y=c_embeddings.unsqueeze(0)  # (1, b, h)
        )

        logits2 = logits1.T

        labels = torch.arange(logits1.size(0)).long().to(device)

        loss1 = self.loss_fn(logits1, labels)
        loss2 = self.loss_fn(logits2, labels)
        loss = (loss1 + loss2) / 2.0

        # ---
        loss_dict = {
            "nce": loss,
        }

        return loss, loss_dict


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Sci-LLM Model
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class SciEncoder(nn.Module):
    """
    The Sci-LLM encoder
    """

    def __init__(self, config):
        print("initializing the Sci-LLM encoder...")

        super(SciEncoder, self).__init__()
        self.config = config

        # ----------------------------- Backbone -----------------------------------------#
        backbone_config = AutoConfig.from_pretrained(self.config["backbone_path"])
        backbone_config.update(
            {
                "use_cache": False,
            }
        )

        self.backbone = AutoModel.from_pretrained(self.config["backbone_path"], config=backbone_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.backbone.resize_token_embeddings(config["len_tokenizer"])

        # enable gradient checkpointing
        if config["gradient_checkpointing"]:
            self.backbone.gradient_checkpointing_enable()

        # freeze embeddings
        if config["n_freeze"] > 0:
            print(f"setting requires grad to false for first {config['n_freeze']} layers")
            self.backbone.embeddings.requires_grad_(False)
            self.backbone.encoder.layer[:config["n_freeze"]].requires_grad_(False)

        # Pooling
        self.pool = MeanPooling()

    def encode(self, batch, prefix="q"):

        outputs = self.backbone(
            input_ids=batch[f"{prefix}_input_ids"],
            attention_mask=batch[f"{prefix}_attention_mask"],
            output_hidden_states=False,
        )

        encoder_layer = outputs.last_hidden_state
        if self.config["pooler"] == "cls":
            embeddings = encoder_layer[:, :1].squeeze(1)
        else:
            embeddings = self.pool(encoder_layer, batch[f"{prefix}_attention_mask"])  # mean pooling
        return embeddings


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

        adv_loss, _ = self.model(batch)
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
