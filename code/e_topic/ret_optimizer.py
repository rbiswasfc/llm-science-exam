import bitsandbytes as bnb
from torch.optim import AdamW


def get_grouped_params(model, config):
    """
    layerwise learning rate decay implementation
    """
    grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": config["lr"],
            "weight_decay": config["weight_decay"]*1e-3,
        },
    ]

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    print("getting params for query encoder...")
    query_layers = [model.query_encoder.backbone.embeddings] + list(model.query_encoder.backbone.encoder.layer)

    print("getting params for content encoder...")
    content_layers = [model.content_encoder.backbone.embeddings] + list(model.content_encoder.backbone.encoder.layer)

    query_layers.reverse()
    lr = config["query_lr"]
    wd = config["query_weight_decay"]
    llrd = config["query_llrd"]

    for layer in query_layers:
        grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": wd,
                "lr": lr,
            },

            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.,
                "lr": lr,
            },
        ]
        lr *= llrd

    # --
    content_layers.reverse()
    lr = config["content_lr"]
    wd = config["content_weight_decay"]
    llrd = config["content_llrd"]

    for layer in content_layers:
        grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": wd,
                "lr": lr,
            },

            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.,
                "lr": lr,
            },
        ]
        lr *= llrd

    return grouped_parameters


def get_optimizer(model, config):
    """
    optimizer for model training
    """
    config = config["optimizer"]
    optimizer_grouped_parameters = get_grouped_params(model, config)

    optimizer_kwargs = {
        "betas": (config["beta1"], config["beta2"]),
        "eps": config['eps'],
        "lr": config["lr"]
    }

    if config["use_bnb"]:
        optimizer = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            **optimizer_kwargs
        )
        return optimizer
    else:
        optimizer = AdamW(
            optimizer_grouped_parameters,
            **optimizer_kwargs
        )

    return optimizer
