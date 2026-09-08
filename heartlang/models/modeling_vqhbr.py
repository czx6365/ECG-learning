# --------------------------------------------------------
# Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model
# By Jiarui Jin and Haoyu Wang
# Based on BEiT-v2, timm, DeiT, DINO and LaBraM code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# https://github.com/935963004/LaBraM
# ---------------------------------------------------------

"""VQHBR tokenizer: 将连续 ECG token 压缩到离散 codebook。"""

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from heartlang.models.backbone_vqhbr import VqhbrBackbone
from heartlang.utils.norm_ema_quantizer import NormEMAVectorQuantizer


class VQHBR(nn.Module):
    """VQ tokenizer 主体,包含 encoder、quantizer 和 decoder。"""
    def __init__(
        self,
        encoder_config,
        decoder_config,
        n_embed=8192,
        codebook_embed_dim=128,
        decay=0.99,
        quantize_kmeans_init=True,
        decoder_out_dim=96,
        smooth_l1_loss=False,
        **kwargs,
    ):
        super().__init__()

        print("Final encoder config", encoder_config)
        self.encoder = VqhbrBackbone(**encoder_config)
        print("Final decoder config", decoder_config)
        self.decoder = VqhbrBackbone(**decoder_config)

        self.quantize = NormEMAVectorQuantizer(
            n_embed=n_embed,
            embedding_dim=codebook_embed_dim,
            beta=1.0,
            kmeans_init=quantize_kmeans_init,
            decay=decay,
            balance_loss_weight=kwargs.get("codebook_balance_loss_weight", 0.0),
            balance_temperature=kwargs.get("codebook_balance_temperature", 0.1),
        )

        self.decoder_out_dim = decoder_out_dim

        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config["embed_dim"], encoder_config["embed_dim"]),
            nn.Tanh(),
            nn.Linear(encoder_config["embed_dim"], codebook_embed_dim),
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config["embed_dim"], decoder_config["embed_dim"]),
            nn.Tanh(),
            nn.Linear(decoder_config["embed_dim"], self.decoder_out_dim),
        )

        self.kwargs = kwargs

        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.polarity_pos_codes = []
        self.polarity_neg_codes = []
        self.polarity_loss_weight = 0.0
        self.polarity_margin = 0.1
        self.polarity_force_assignment = False

    def configure_polarity_pairs(
        self,
        pairs=None,
        loss_weight=0.0,
        margin=0.1,
        force_assignment=False,
    ):
        pairs = pairs or []
        self.polarity_pos_codes = [int(pos_code) for pos_code, _ in pairs]
        self.polarity_neg_codes = [int(neg_code) for _, neg_code in pairs]
        self.polarity_loss_weight = float(loss_weight)
        self.polarity_margin = float(margin)
        self.polarity_force_assignment = bool(force_assignment)

    def set_polarity_force_assignment(self, enabled):
        self.polarity_force_assignment = bool(enabled)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    @property
    def device(self):
        return self.decoder.cls_token.device

    def get_number_of_tokens(self):
        return self.quantize.num_tokens

    def get_tokens(self, data, in_chan_matrix=None, in_time_matrix=None, **kwargs):
        quantize, embed_ind, loss = self.encode(
            data, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix
        )
        output = {}
        output["token"] = embed_ind.view(data.shape[0], -1)
        output["input_img"] = data
        output["quantize"] = quantize

        return output

    def encode_to_quantizer_features(self, x, in_chan_matrix=None, in_time_matrix=None):
        bs, seq_len, time_window = x.shape
        encoder_features = self.encoder(
            x,
            in_chan_matrix=in_chan_matrix,
            in_time_matrix=in_time_matrix,
            return_qrs_tokens=True,
        )
        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(
                encoder_features.type_as(self.encode_task_layer[-1].weight)
            )
        return F.normalize(to_quantizer_features, p=2, dim=-1)

    def encode(
        self,
        x,
        in_chan_matrix=None,
        in_time_matrix=None,
        valid_token_mask=None,
        polarity_labels=None,
    ):
        to_quantizer_features = self.encode_to_quantizer_features(
            x, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix
        )
        quantize, loss, embed_ind = self.quantize(
            to_quantizer_features,
            valid_token_mask=valid_token_mask,
            polarity_labels=polarity_labels,
            polarity_pos_codes=self.polarity_pos_codes,
            polarity_neg_codes=self.polarity_neg_codes,
            polarity_loss_weight=self.polarity_loss_weight,
            polarity_margin=self.polarity_margin,
            polarity_force_assignment=self.polarity_force_assignment,
        )
        return quantize, embed_ind, loss

    def decode(self, quantize, in_chan_matrix=None, in_time_matrix=None, **kwargs):
        decoder_features = self.decoder(
            quantize,
            in_chan_matrix=in_chan_matrix,
            in_time_matrix=in_time_matrix,
            return_qrs_tokens=True,
        )
        rec = self.decode_task_layer(decoder_features)
        return rec

    def get_codebook_indices(
        self, x, in_chan_matrix=None, in_time_matrix=None, **kwargs
    ):
        return self.get_tokens(
            x, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix, **kwargs
        )["token"]

    def get_codebook_quantize(
        self, x, in_chan_matrix=None, in_time_matrix=None, **kwargs
    ):
        return self.get_tokens(
            x, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix, **kwargs
        )["quantize"]

    def calculate_rec_loss(self, rec, target, valid_token_mask=None):
        if valid_token_mask is None:
            rec_loss = self.loss_fn(rec, target)
        elif valid_token_mask.any():
            if self.loss_fn is F.smooth_l1_loss:
                rec_loss = F.smooth_l1_loss(rec[valid_token_mask], target[valid_token_mask])
            else:
                rec_loss = F.mse_loss(rec[valid_token_mask], target[valid_token_mask])
        else:
            rec_loss = rec.sum() * 0.0
        return rec_loss

    def compute_polarity_labels(self, x):
        # 与评估脚本一致：用 heartbeat 中绝对幅值最大的峰值符号判断正/负极性。
        peak_idx = torch.argmax(torch.abs(x), dim=-1, keepdim=True)
        peak_value = torch.gather(x, dim=-1, index=peak_idx).squeeze(-1)
        return torch.where(peak_value >= 0, torch.ones_like(peak_value), -torch.ones_like(peak_value)).long()

    def forward(self, x, in_chan_matrix=None, in_time_matrix=None, **kwargs):
        """
        x: [bs, seq_len, time_window]
        """
        valid_token_mask = None
        if in_chan_matrix is not None and in_time_matrix is not None:
            valid_token_mask = (
                (in_chan_matrix > 0)
                & (in_time_matrix > 0)
                & ~(x == 0).all(dim=-1)
            )

        polarity_labels = None
        if self.polarity_pos_codes and self.polarity_loss_weight > 0:
            polarity_labels = self.compute_polarity_labels(x)

        quantize, embed_ind, emb_loss = self.encode(
            x,
            in_chan_matrix,
            in_time_matrix,
            valid_token_mask=valid_token_mask,
            polarity_labels=polarity_labels,
        )
        xrec = self.decode(quantize, in_chan_matrix, in_time_matrix)

        rec_loss = self.calculate_rec_loss(xrec, x, valid_token_mask=valid_token_mask)
        loss = emb_loss + rec_loss

        log = {}
        split = "train" if self.training else "val"
        log[f"{split}/quant_loss"] = emb_loss.detach().mean()
        log[f"{split}/rec_loss"] = rec_loss.detach().mean()
        log[f"{split}/total_loss"] = loss.detach().mean()
        balance_loss = getattr(self.quantize, "last_balance_loss", None)
        usage_perplexity = getattr(self.quantize, "last_usage_perplexity", None)
        if balance_loss is not None:
            log[f"{split}/balance_loss"] = balance_loss.detach().mean()
        if usage_perplexity is not None:
            log[f"{split}/batch_usage_perplexity"] = usage_perplexity.detach().mean()
        polarity_loss = getattr(self.quantize, "last_polarity_loss", None)
        polarity_pair_count = getattr(self.quantize, "last_polarity_pair_count", None)
        polarity_assignment_acc = getattr(self.quantize, "last_polarity_assignment_acc", None)
        polarity_forced_count = getattr(self.quantize, "last_polarity_forced_count", None)
        if polarity_loss is not None:
            log[f"{split}/polarity_loss"] = polarity_loss.detach().mean()
        if polarity_pair_count is not None:
            log[f"{split}/polarity_pair_count"] = polarity_pair_count.detach().mean()
        if polarity_assignment_acc is not None:
            log[f"{split}/polarity_assignment_acc"] = polarity_assignment_acc.detach().mean()
        if polarity_forced_count is not None:
            log[f"{split}/polarity_forced_count"] = polarity_forced_count.detach().mean()

        return loss, log


def get_model_default_params():
    return dict(
        seq_len=256,
        time_window=96,
        embed_dim=768,
        depth=4,
        heads=2,
        mlp_dim=2048,
        dropout=0.0,
        emb_dropout=0.0,
    )


@register_model
def vqhbr(
    pretrained=False,
    pretrained_weight=None,
    as_tokenzer=False,
    n_code=8192,
    code_dim=128,
    **kwargs,
):
    encoder_config, decoder_config = (
        get_model_default_params(),
        get_model_default_params(),
    )

    encoder_config["depth"] = 4
    encoder_config["heads"] = 2
    encoder_config["Encoder"] = True
    decoder_config["code_dim"] = code_dim
    decoder_config["depth"] = 2
    decoder_config["heads"] = 2
    decoder_config["Encoder"] = False
    decoder_out_dim = 96

    kwargs["model"] = "vqhbr"

    model = VQHBR(
        encoder_config,
        decoder_config,
        n_code,
        code_dim,
        decoder_out_dim=decoder_out_dim,
        **kwargs,
    )

    if as_tokenzer:
        assert pretrained
        assert pretrained_weight is not None

        if pretrained_weight.startswith("https"):
            weights = torch.hub.load_state_dict_from_url(
                pretrained_weight, map_location="cpu", check_hash=True
            )
        else:
            weights = torch.load(
                pretrained_weight, map_location="cpu", weights_only=False
            )

        if "model" in weights:
            weights = weights["model"]
        else:
            weights = weights["state_dict"]
        keys = list(weights.keys())

        for k in keys:
            if (
                k.startswith("loss")
                or k.startswith("teacher")
                or k.startswith("scaling")
            ):
                del weights[k]
        model.load_state_dict(weights)
    return model


if __name__ == "__main__":
    x = torch.randn(4, 256, 96)
    model = vqhbr()
    out = model(x)
    print(f"output: {out}")
