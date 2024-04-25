import torch
import torch.distributed
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.distributed as dist
import common


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(
        tensor.size()[last_dim], num_partitions
    )  # 得到每个切分的size
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)  # 对张量进行切分
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if common.get_val("G_TENSOR_PARALLEL_GROUP_WORLD_SIZE") == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(
        input_, group=common.get_val("G_TENSOR_PARALLEL_GROUP")
    )

    return input_


def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = common.get_val(
        "G_TENSOR_PARALLEL_GROUP_WORLD_SIZE"
    )  # 获取本tensor进程组的world size
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = dist.get_rank() // 2  # 获取自己的rank
    output = input_list[rank].contiguous()  # 获取切分后，自己对应的tensor

    return output


def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = common.get_val("G_TENSOR_PARALLEL_GROUP_WORLD_SIZE")
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = dist.get_rank() // 2  # 获得本worker在tensor并行之中的rank

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    # 在本 tensor 进程组之间进行 all-gather操作
    torch.distributed.all_gather(
        tensor_list, input_, group=common.get_val("G_TENSOR_PARALLEL_GROUP")
    )
    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_  # 简单的把输入转移到输出，就是对应了前向复制identity

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时候，输入是多个GPU上的梯度整体，通过all-reduce合并
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


# -----------------
# Helper functions.
# -----------------


def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


# ------end--------
# Helper functions.
# -----------------
class ColumnParallelLinear(torch.nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        gather_output=True,
        init_data=None,
    ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = common.get_val(
            "G_TENSOR_PARALLEL_GROUP_WORLD_SIZE"
        )  # 获得本tensor并行组的world size
        self.output_size_per_partition = divide(
            output_size, world_size
        )  # 获得本子模型应输出size

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        if init_data is None:
            self.weight = Parameter(
                torch.empty(  # 用切分的size初始化权重
                    self.output_size_per_partition,
                    self.input_size,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
            )
            init.constant_(self.weight, 1.0)
        else:
            rank = dist.get_rank() // 2  # 获取自己的rank
            row, col = init_data.shape[0], init_data.shape[1]
            self.weight = Parameter(
                init_data[rank * row // 2 : (rank + 1) * row // 2, :],
            )
        if bias:
            self.bias = Parameter(
                torch.empty(  # 用切分的size初始化权重
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
            )
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def forward(self, input_):
        # Set up backprop all-reduce.
        # 建立反向传播all-reduce，就是图中f的backward
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.

        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        # 下面就是图中的 g 操作
        if self.gather_output:
            # All-gather across the partitions.
            # 聚合输出，就是图中g的forward
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output


class RowParallelLinear(torch.nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=False,
        init_data=None,
    ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = common.get_val("G_TENSOR_PARALLEL_GROUP_WORLD_SIZE")
        self.input_size_per_partition = divide(input_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.

        # 切分权重
        if init_data is None:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
            )
            init.constant_(self.weight, 1.0)
        else:
            rank = dist.get_rank() // 2  # 获取自己的rank
            row, col = init_data.shape[0], init_data.shape[1]
            self.weight = Parameter(
                init_data[:, rank * col // 2 : (rank + 1) * col // 2],
            )
        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.output_size,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
            )
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def forward(self, input_):
        # Set up backprop all-reduce.
        # 这里，输入的张量已经被分割到每个GPU，输出张量是all-reduce之后的整体
        if self.input_is_parallel:  # 是否已经是split的输入
            # Transformers's MLP 到达这里，因为已经split，所以直接就接了输入，不会scatter
            input_parallel = input_
        else:
            # 独立 row parallel 线性层 到了这里，需要自行进行前向切分和后向拼接
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        # X_i 和 A_i
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        # 进行前向all-reduce操作，这样每个GPU之上都是完整的最新结果，同时搭建了后向的identity操作
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        output = output_ + self.bias if self.bias is not None else output_
        return output


class ParallelMLP(torch.nn.Module):

    def __init__(self, input_size, output_size, originMLP):
        super(ParallelMLP, self).__init__()
        self.up_proj = ColumnParallelLinear(
            input_size=input_size,
            output_size=output_size,
            bias=False,
            gather_output=False,
            init_data=originMLP.up_proj.weight.data,
        )
        self.gate_proj = ColumnParallelLinear(
            input_size=input_size,
            output_size=output_size,
            bias=False,
            gather_output=False,
            init_data=originMLP.gate_proj.weight.data,
        )
        self.donw_proj = RowParallelLinear(
            input_size=output_size,
            output_size=input_size,
            bias=False,
            input_is_parallel=True,
            init_data=originMLP.down_proj.weight.data,
        )
        self.act_fn = F.silu

    def forward(self, inp):
        return self.donw_proj(F.silu(self.gate_proj(inp)) * self.up_proj(inp))


class ParallelAttention(torch.nn.Module):

    def __init__(self) -> None:
        super(ParallelAttention, self).__init__()

    def forward(self, inp):
        pass
