import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# launch command: torchrun --standalone --nnodes=1 --nproc-per-node=4 test_tp.py
import torch.distributed as dist
from torch.distributed._tensor import (
    Shard,
    Replicate,
    DTensor,
    distribute_tensor,
    distribute_module,
    init_device_mesh,
    DeviceMesh,
)
from torch.distributed.tensor.parallel import (
    PairwiseParallel,
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.config = config
        # self.hidden_size = config.hidden_size
        # self.intermediate_size = config.intermediate_size
        self.config = None
        self.hidden_size = 4096
        self.intermediate_size = 11008
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def print0(msg):
    if dist.get_rank() == 0:
        print(msg)


mesh = DeviceMesh("cuda", [[0, 1, 2, 3], [4, 5, 6, 7]], mesh_dim_names=["dp", "tp"])
rank = dist.get_rank()
torch.manual_seed(42)
inp = torch.randn(1, 4096)
model = LlamaMLP(config=None).to("cuda")
plan = {
    "gate_proj": ColwiseParallel(),
    "up_proj": ColwiseParallel(),
    "down_proj": RowwiseParallel(),
}


out1 = model(inp.to("cuda"))
sharded_module = parallelize_module(model, mesh, plan)
out2 = sharded_module(inp.to("cuda"))
# print0(out1.sum())
# print0(out2.sum())
for name, child in model.named_children():
    print(
        f"rank {rank}----> module name: {name}\t module weight {child.weight.shape}\t module local weight {child.weight.to_local().shape}\t module weight deivce {child.weight.device}"
    )
    break
