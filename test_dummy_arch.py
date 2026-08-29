import torch
import utils
from unittest.mock import MagicMock

utils.Decode = MagicMock

from slr_network import SLRModel

B, T = 2, 32
num_classes = 64
gloss_dict = {f"g{i}": [i, 10] for i in range(1, 64)}

model = SLRModel(
    num_classes=num_classes,
    c2d_type="resnet18",
    conv_type=2,
    use_bn=True,
    hidden_size=1024,
    gloss_dict=gloss_dict,
    loss_weights={"SeqCTC": 1.0},
    weight_norm=True,
    share_classifier=True,
    use_attention=True,
    attention_heads=4,
)
model.train()

x = torch.randn(B, T, 3, 224, 224)
len_x = torch.LongTensor([T, T - 4])
label = torch.LongTensor([1, 5, 10, 2, 3])
label_lgt = torch.LongTensor([3, 2])

print("=== Forward Pass ===")
ret = model(x, len_x, label, label_lgt)
for k, v in ret.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")
    else:
        print(f"  {k}: {type(v).__name__}")

loss = model.criterion_calculation(ret, label, label_lgt)
print(f"\n=== Loss ===")
print(f"  SeqCTC Loss: {loss.item():.4f}")

loss.backward()
print(f"\n=== Backward ===")
print(f"  Gradient OK!")

print(f"\n=== Attention ===")
if hasattr(model, 'temporal_attention'):
    print(f"  TemporalSelfAttention: AKTIF")
    for name, p in model.temporal_attention.named_parameters():
        print(f"  {name}: grad={'ada' if p.grad is not None else 'None'}")
else:
    print(f"  Attention TIDAK aktif")

print("\nDummy test PASSED!")
