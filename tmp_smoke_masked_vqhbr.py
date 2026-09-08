import torch
from timm.models import create_model

from heartlang.models import modeling_vqhbr  # noqa: F401


model = create_model(
    "vqhbr",
    pretrained=False,
    as_tokenzer=False,
    n_code=32,
    code_dim=8,
    quantize_kmeans_init=False,
)
model.train()
x = torch.randn(2, 256, 96)
ch = torch.ones(2, 256, dtype=torch.long)
tm = torch.ones(2, 256, dtype=torch.long)
x[:, 100:, :] = 0
ch[:, 100:] = 0
tm[:, 100:] = 0
loss, log = model(x, in_chan_matrix=ch, in_time_matrix=tm)
expected_valid = int(((ch > 0) & (tm > 0) & ~(x == 0).all(dim=-1)).sum())
print("loss_finite", bool(torch.isfinite(loss).item()))
print("loss", float(loss.detach()))
print("cluster_size_sum", float(model.quantize.cluster_size.sum()))
print("expected_valid", expected_valid)
print("log_keys", sorted(log))
