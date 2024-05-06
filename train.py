import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.checkpoint import checkpoint_sequential

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model

from layers import ColumnParallelLinear, RowParallelLinear, ParallelMLP
import common

ARG_G = None
common._init()


def get_args():
    global ARG_G
    return ARG_G


device_map = [
    {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.2": 0,
        "model.layers.3": 0,
        "model.layers.4": 0,
        "model.layers.5": 0,
        "model.layers.6": 0,
        "model.layers.7": 0,
        "model.layers.8": 0,
        "model.layers.9": 0,
        "model.layers.10": 0,
        "model.layers.11": 0,
        "model.layers.12": 0,
        "model.layers.13": 0,
        "model.layers.14": 0,
        "model.layers.15": 0,
        "model.layers.16": 1,
        "model.layers.17": 1,
        "model.layers.18": 1,
        "model.layers.19": 1,
        "model.layers.20": 1,
        "model.layers.21": 1,
        "model.layers.22": 1,
        "model.layers.23": 1,
        "model.layers.24": 1,
        "model.layers.25": 1,
        "model.layers.26": 1,
        "model.layers.27": 1,
        "model.layers.28": 1,
        "model.layers.29": 1,
        "model.layers.30": 1,
        "model.layers.31": 1,
        "model.norm": 1,
        "lm_head": 1,
    },
    {
        "model.embed_tokens": 2,
        "model.layers.0": 2,
        "model.layers.1": 2,
        "model.layers.2": 2,
        "model.layers.3": 2,
        "model.layers.4": 2,
        "model.layers.5": 2,
        "model.layers.6": 2,
        "model.layers.7": 2,
        "model.layers.8": 2,
        "model.layers.9": 2,
        "model.layers.10": 2,
        "model.layers.11": 2,
        "model.layers.12": 2,
        "model.layers.13": 2,
        "model.layers.14": 2,
        "model.layers.15": 2,
        "model.layers.16": 3,
        "model.layers.17": 3,
        "model.layers.18": 3,
        "model.layers.19": 3,
        "model.layers.20": 3,
        "model.layers.21": 3,
        "model.layers.22": 3,
        "model.layers.23": 3,
        "model.layers.24": 3,
        "model.layers.25": 3,
        "model.layers.26": 3,
        "model.layers.27": 3,
        "model.layers.28": 3,
        "model.layers.29": 3,
        "model.layers.30": 3,
        "model.layers.31": 3,
        "model.norm": 3,
        "lm_head": 3,
    },
]


def cleanup():
    dist.destroy_process_group()


tokenizer = AutoTokenizer.from_pretrained(
    "/home/genghaozhe/workspace/huggingface-models/meta-llama/Llama-2-7b-hf"
)
tokenizer.pad_token_id = 2
tokenizer.padding_side = "right"


def get_model():
    rank = dist.get_rank()
    model = LlamaForCausalLM.from_pretrained(
        "/home/genghaozhe/workspace/huggingface-models/meta-llama/Llama-2-7b-hf",
        low_cpu_mem_usage=True,
        # device_map=device_map[rank],
        # device_map=rank,
        device_map="cpu",
    )
    return model


def process_func(example):
    MAX_LENGTH = 384  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        "\n".join(["Human: " + example["instruction"], example["input"]]).strip()
        + "\n\nAssistant: ",
        add_special_tokens=False,
    )
    response = tokenizer(example["output"], add_special_tokens=False)
    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    )
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.eos_token_id]
    )
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def get_dataset():
    ds = load_dataset("shibing624/alpaca-zh", split="train")
    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
    return tokenized_ds


def lora_config_model(model):
    print_rank_0(f"get the model with lora")
    config = LoraConfig(task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, config)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    # model = model.half()  # 当整个模型都是半精度时，需要将adam_epsilon调大
    model.print_trainable_parameters()
    return model


def print_rank_0(content):
    if dist.get_rank() == 0:
        print(content)


def initialize_parallel_group():
    G_DATA_PARALLEL = dist.new_group([0, 1])
    common.set_val("G_DATA_PARALLEL", G_DATA_PARALLEL)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tensor_parallel_size = 2
    for i in range(2):
        # i = 0, 1
        ranks = range(i, world_size, 2)
        print_rank_0(list(ranks))
        group = dist.new_group(ranks)
        if rank in ranks:
            common.set_val("G_TENSOR_PARALLEL_GROUP", group)
            common.set_val("G_TENSOR_PARALLEL_GROUP_WORLD_SIZE", 2)
            G_TENSOR_PARALLEL_GROUP = group
            G_TENSOR_PARALLEL_GROUP_WORLD_SIZE = 2


def test_parallel():
    torch.manual_seed(42)
    inp = torch.rand(1, 2)

    linear1 = nn.Linear(2, 4, bias=False)
    linear3 = nn.Linear(2, 4, bias=False)
    linear2 = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        linear1.weight.fill_(1.0)
        linear2.weight.fill_(1.0)
        linear3.weight.fill_(1.0)
    # serial_mlp = nn.Sequential(linear1, linear2)
    # serial_out = serial_mlp(inp)
    serial_out = linear2(linear1(inp) * linear3(inp))

    p_linear1 = ColumnParallelLinear(
        2, 4, bias=False, gather_output=False, init_data=linear1.weight.data
    ).to("cuda")
    p_linear3 = ColumnParallelLinear(
        2, 4, bias=False, gather_output=False, init_data=linear3.weight.data
    ).to("cuda")
    p_linear2 = RowParallelLinear(
        4, 2, bias=False, input_is_parallel=True, init_data=linear2.weight.data
    ).to("cuda")
    rank = dist.get_rank()
    # parallel_mlp = nn.Sequential(p_linear1, p_linear2).to("cuda")
    # parallel_out = parallel_mlp(inp.to("cuda"))
    parallel_out = p_linear2(p_linear1(inp.to("cuda")) * p_linear3(inp.to("cuda")))

    print_rank_0(
        f"rank: {dist.get_rank()}\nparallel output: {parallel_out}\nserial output: {serial_out}"
    )


def test_mlp(layers):
    torch.manual_seed(42)
    inp = torch.rand(1, 4096)
    mlp = layers[0].mlp
    print_rank_0(mlp)
    gate_proj = mlp.gate_proj
    up_proj = mlp.up_proj
    down_proj = mlp.down_proj
    mlp_out = mlp.to("cuda")(inp.to("cuda"))

    # gate_proj_p = ColumnParallelLinear(
    #     4096, 11008, bias=False, gather_output=False, init_data=gate_proj.weight.data
    # ).to("cuda")
    # up_proj_p = ColumnParallelLinear(
    #     4096, 11008, bias=False, gather_output=False, init_data=up_proj.weight.data
    # ).to("cuda")
    # down_proj_p = RowParallelLinear(
    #     11008,
    #     4096,
    #     bias=False,
    #     input_is_parallel=True,
    #     init_data=down_proj.weight.data,
    # ).to("cuda")
    # out = F.silu(gate_proj_p(inp.to("cuda")))
    # out = up_proj_p(inp.to("cuda")) * out
    # parallel_out = down_proj_p(out)

    parallel_mlp = ParallelMLP(4096, 11008, mlp).to("cuda")
    parallel_out = parallel_mlp(inp.to("cuda"))

    print_rank_0(f"mlp out: {mlp_out.sum()}, parallel out: {parallel_out.sum()}")
    print_rank_0(f'diff: {(mlp_out.to("cuda") - parallel_out).sum()}')


def convert_mlp(layers):
    for l in layers:
        l.mlp = ParallelMLP(4096, 11008, l.mlp)  # .to("cuda")


def test_attn(layers):
    torch.manual_seed(42)
    inp = torch.rand(1, 4096)
    attn = layers[0].self_attn
    print_rank_0(attn)


def convert_attn(layers):
    self_attn = layers[0].self_attn
    print_rank_0(self_attn)


def tensor_parallelize_model(model):
    initialize_parallel_group()
    decoder_layers = model.model.layers  # ModuleList[ (self_attn, mlp) * 32]
    # test_parallel()
    # test_mlp(decoder_layers)
    convert_mlp(decoder_layers)
    # test_attn(decoder_layers)
    # convert_attn(decoder_layers)
    # print_rank_0(model)
    return model


def _run_worker():
    dist.init_process_group("nccl")
    # dp_pg = dist.new_group([0, 4])
    # sub_pg = dist.new_subgroups(group_size=2)
    args = get_args()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    world_size = dist.get_world_size()
    if rank in (2, 3):
        model = get_model()
        if args.use_tp:
            print_rank_0("Use tensor parallel.")
            model = tensor_parallelize_model(model)

    if rank in (0, 1):
        model = get_model()
        # model = None
        if args.use_tp:
            print_rank_0("Use tensor parallel.")
            model = tensor_parallelize_model(model)

        # return
        model = lora_config_model(model).to("cuda")

        if args.use_amp:
            print_rank_0("Use auto mixed precision training.")

        if args.use_grad_ckpt:
            # model.base_model.model.model.gradient_checkpointing = True
            model.base_model.model.gradient_checkpointing_enable()
            print_rank_0("Use gradient checkpoint.")

        if args.use_dp:
            model = DDP(
                model, process_group=common.get_val("G_DATA_PARALLEL")
            )  # dist.new_group([0, 1]))
            print_rank_0("Use distributed data parallel.")

        # loss_fn = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, eps=1e-4)
        tokenized_ds = get_dataset()
        # ds_6k = tokenized_ds.select(range(3000))
        ds_6k = tokenized_ds.select(range(32))  # for debug
        sampler = DistributedSampler(ds_6k, num_replicas=2) if args.use_dp else None
        dl = DataLoader(
            ds_6k,
            batch_size=16,
            collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
            sampler=sampler,
        )
        print_rank_0(f"get the dataloader")
        model.train()
        num_epoch = 1
        log_interval = 1
        scaler = amp.GradScaler()
        for epoch in range(num_epoch):
            print_rank_0(f"start train")
            if args.use_dp:
                sampler.set_epoch(epoch)
            for batch_idx, data in enumerate(dl):
                if args.use_amp:
                    with torch.autocast(
                        device_type="cuda", dtype=torch.float16, enabled=True
                    ):
                        print_rank_0(f"FWD 111")
                        output = model(**data.to(f"cuda:{rank}"))
                        print_rank_0(f"FWD 222")
                    scaler.scale(output.loss).backward()
                    # optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(**data.to("cuda"))
                    output.loss.backward()
                    optimizer.step()

                optimizer.zero_grad()
                if batch_idx % log_interval == 0:
                    print_rank_0(
                        f"Step: {batch_idx}\t Data: {data['input_ids'].shape}\t Training Loss: {output.loss.item()}"
                    )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
                                    A demo for llama fine-tuning with gradient checkpoint, 
                                    mixed precision, data parallel, tensor parallel.
                                    """
    )
    parser.add_argument("--use_dp", help="use data parallel", action="store_true")
    parser.add_argument("--use_tp", help="use tensor parallel", action="store_true")
    parser.add_argument("--use_amp", help="use mixed precesion", action="store_true")
    parser.add_argument(
        "--use_grad_ckpt", help="use gradient checkpoint", action="store_true"
    )
    ARG_G = parser.parse_args()
    _run_worker()
    # torchrun --standalone  --nnodes=1 --nproc_per_node=2  train.py --use_dp --use_amp --use_grad_ckpt
    # torchrun --standalone  --nnodes=1 --nproc_per_node=2  train.py --use_dp --use_amp --use_grad_ckpt --use_tp
    # torchrun --standalone  --nnodes=1 --nproc_per_node=4  train.py --use_dp --use_grad_ckpt --use_amp --use_tp > train.log 2>&1
