# launch command: torchrun --standalone --nnodes=1 --nproc-per-node=4 test_tp.py
import torch
import torch.nn as nn
import torch.optim as optim
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


def print0(msg, rank):
    if rank == 0:
        print(msg)


# ================================== Demo 1 =====================================
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, 8)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.fc1(input) + self.fc2(input))


# mesh = init_device_mesh("cuda", (2,))
mesh = DeviceMesh("cuda", [2, 3])


def shard_params(mod_name, mod, mesh):
    col_linear_placement = [Shard(0)]
    # shard fc1 and fc2
    if isinstance(mod, nn.Linear):
        for name, param in mod.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(param, mesh, col_linear_placement)
            )
            mod.register_parameter(name, dist_param)


model = MyModule()
sharded_module = distribute_module(model, mesh, partition_fn=shard_params)
# for name, param in model.named_parameters():
#     # print0(f"{name}\n{param}", rank=dist.get_rank())
#     if dist.get_rank() == 2 or dist.get_rank() == 3:
#         print(f"{name}\n{param.shape}")
"""output
DTensor(local_tensor=tensor([]), device_mesh=DeviceMesh:([2, 3]), placements=(Shard(dim=0),))
DTensor(local_tensor=tensor([]), device_mesh=DeviceMesh:([2, 3]), placements=(Shard(dim=0),))
DTensor(local_tensor=tensor([[ 0.1630, -0.2858,  0.3327,  0.0254,  0.0051, -0.2719, -0.2348,  0.1766],
        [-0.0711,  0.1479,  0.2841, -0.1510, -0.0669,  0.3317,  0.2552, -0.2576]],
       device='cuda:2'), device_mesh=DeviceMesh:([2, 3]), placements=(Shard(dim=0),))
DTensor(local_tensor=tensor([[-0.1100,  0.3219,  0.0787, -0.2790,  0.2037, -0.0205,  0.1215,  0.2251],
        [-0.2637, -0.2776,  0.3445,  0.2071,  0.0217,  0.0756,  0.2859,  0.2113]],
       device='cuda:3'), device_mesh=DeviceMesh:([2, 3]), placements=(Shard(dim=0),))
"""


# ================================== Demo 2: shard =====================================
def shard_big_tensor(world_size):
    mesh = DeviceMesh("cuda", [0, 1, 2, 3])
    big_tensor = torch.arange(32).reshape(4, 8)
    dtensor = distribute_tensor(big_tensor, mesh, [Shard(0)])
    print(
        f"on rank: {dist.get_rank()}, dtensor global shape: {dtensor.shape}, local shape: {dtensor.to_local().shape}."
    )


# shard_big_tensor(dist.get_world_size())


# ================================== Demo 3: replicate =====================================
def replicate_big_tensor(world_size):
    mesh = DeviceMesh("cuda", [0, 1, 2, 3])
    big_tensor = torch.arange(32).reshape(4, 8)
    dtensor = distribute_tensor(big_tensor, mesh, [Replicate()])
    print(
        f"on rank: {dist.get_rank()}, dtensor global shape: {dtensor.shape}, local shape: {dtensor.to_local().shape}."
    )


# replicate_big_tensor(dist.get_world_size())


# ================================== Demo 4: partial =====================================
def partial_big_tensor(world_size):
    mesh = DeviceMesh("cuda", [[0, 1], [2, 3]])
    big_tensor = torch.arange(32).reshape(4, 8)
    spec = [Replicate(), Shard(0)]
    dtensor = distribute_tensor(big_tensor, mesh, placements=spec)
    print(
        f"on rank: {dist.get_rank()}, dtensor global shape: {dtensor.shape}, local shape: {dtensor.to_local().shape}."
    )


# partial_big_tensor(dist.get_world_size())


# ================================== Demo 5: to_local, from_local =====================================
# 注意：from_local 和 to_local都是可微分的，也就是说可以用在神经网络nn.Module中
def dtensor_from_local_to_local(world_size):
    mesh = DeviceMesh("cuda", [0, 1, 2, 3])
    local_tensor = torch.arange(32, requires_grad=True).reshape(4, 8)
    rowwise_placement = [Shard(0)]
    rowwise_tensor = DTensor.from_local(
        local_tensor=local_tensor, device_mesh=mesh, placements=rowwise_placement
    )
    print(
        f"on rank: {dist.get_rank()}, dtensor global shape: {rowwise_tensor.shape}, local shape: {rowwise_tensor.to_local().shape}."
    )


# dtensor_from_local_to_local(dist.get_world_size())


# ================================== Demo 6: reshard =====================================
def dtensor_reshard(world_size):
    local_tensor = torch.rand(8, 8)
    mesh = DeviceMesh("cuda", [0, 1, 2, 3])

    rowwise_placement = [Shard(0)]
    rowwise_tensor = DTensor.from_local(
        local_tensor=local_tensor, device_mesh=mesh, placements=rowwise_placement
    )

    colwise_placement = [Shard(1)]
    colwise_tensor = rowwise_tensor.redistribute(mesh, colwise_placement)
    print(
        f"on rank: {dist.get_rank()}, col-wise dtensor global shape: {colwise_tensor.shape}, local shape: {colwise_tensor.to_local().shape}."
    )

    replica_placement = [Replicate()]
    replica_tensor = colwise_tensor.redistribute(mesh, replica_placement)
    print(
        f"on rank: {dist.get_rank()}, replicate dtensor global shape: {replica_tensor.shape}, local shape: {replica_tensor.to_local().shape}."
    )


# dtensor_reshard(dist.get_world_size())


# ================================== Demo 6: Tensor Parallel Example =====================================


ITER_TIME = 1


class ToyModel(nn.Module):

    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(4, 8)
        self.net0 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(8, 5)

    def forward(self, x):
        return self.net2(self.net1(x) * self.relu(self.net0(x)))


def demo_tp(world_size):
    rank = dist.get_rank()
    print0("Creating a sharding plan based on the given world size", rank)
    device_mesh = DeviceMesh("cuda", [[0, 1], [2, 3]])
    # device_mesh = DeviceMesh("cpu", [0,1,2,3])
    # torch.manual_seed(42)
    model = ToyModel().to("cuda")
    LR = 0.25
    optimizer = optim.SGD(model.parameters(), lr=LR)
    print0("Parallelize the module based on the given ParallelStyle", rank)
    plan = {
        "net1": ColwiseParallel(),
        "net0": ColwiseParallel(),
        "net2": RowwiseParallel(),
    }
    model = parallelize_module(model, device_mesh, parallelize_plan=plan)
    print0(
        f"dtensor: {model.net1.weight.shape}, local tensor: {model.net1.weight.to_local().shape}",
        rank,
    )
    print0(
        f"rank: {rank}\tdtensor: {model.net2.weight.shape}, local tensor: {model.net2.weight.to_local().shape}",
        rank,
    )

    print(f"dtensor: {model.net1.weight}, local tensor: {model.net1.weight.to_local()}")
    for i in range(ITER_TIME):
        inp = torch.randn(2, 4) * rank

        output = model(inp.to("cuda"))
        # print0(f"FWD Step: iter {i}\tinp: {inp}\t out: {output}", rank)
        print(f"rank: {rank}\tFWD Step: iter {i}\tinp: {inp}\t out: {output}")
        sum = output.sum()
        sum.backward()
        # print0(f"BWD Step: iter {i}\toutput sum: {sum}", rank)
        print(f"rank: {rank}\tBWD Step: iter {i}\toutput sum: {sum}")
        optimizer.step()
        # print0(f"Optimization Step: iter {i}", rank)
        optimizer.zero_grad()

    print0("Training finished", rank)


# demo_tp(4)
