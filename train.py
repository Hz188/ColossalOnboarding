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
    print_rank_0(f"====================get the dataset====================")
    ds = load_dataset("shibing624/alpaca-zh", split="train")
    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
    return tokenized_ds


def lora_config_model(model):
    print_rank_0(f"====================get the model with lora====================")
    config = LoraConfig(task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, config)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    # model = model.half()  # 当整个模型都是半精度时，需要将adam_epsilon调大
    if dist.get_rank() == 0:
        model.print_trainable_parameters()
    return model


def print_rank_0(content):
    if dist.get_rank() == 0:
        print(content)


def print_t(content):
    print(f"rank: {dist.get_rank()}\t{content}")


def initialize_parallel_group():
    print_rank_0(
        f"====================initialize the distributed env===================="
    )
    rank = dist.get_rank()  # 0, 1, 2, 3
    world_size = dist.get_world_size()
    for i in range(2):
        # i 0, 1 -> (0, 2) (2, 4)
        start_rank = i * 2
        end_rank = (i + 1) * 2
        for j in range(2):
            # j = 0, 1
            ranks = range(start_rank + j, end_rank, 1)  # (0, 1), (2, 3)
            group = dist.new_group(ranks)
            if rank in ranks and len(ranks) == 2:
                # print(f"dp rank: {rank}-->{list(ranks)}")
                common.set_val("G_DATA_PARALLEL_GROUP", group)
                common.set_val("G_DATA_PARALLEL_GROUP_WORLD_SIZE", 2)
                G_DATA_PARALLEL_GROUP = group
                G_DATA_PARALLEL_GROUP_WORLD_SIZE = 2

    for i in range(2):
        # i = 0, 1
        ranks = range(i, world_size, 2)  # 0 2, 1 3
        group = dist.new_group(ranks)
        if rank in ranks:
            # print(f"tp rank: {rank}-->{list(ranks)}")
            common.set_val("G_TENSOR_PARALLEL_GROUP", group)
            common.set_val("G_TENSOR_PARALLEL_GROUP_WORLD_SIZE", 2)
            G_TENSOR_PARALLEL_GROUP = group
            G_TENSOR_PARALLEL_GROUP_WORLD_SIZE = 2


def convert_mlp(layers):
    print_rank_0(
        f"====================convert origin mlp to parallel mlp===================="
    )
    for l in layers:
        l.mlp = ParallelMLP(4096, 11008, l.mlp)


def test_attn(layers):
    torch.manual_seed(42)
    inp = torch.rand(1, 4096)
    attn = layers[0].self_attn
    print_rank_0(attn)


def convert_attn(layers):
    self_attn = layers[0].self_attn
    print_rank_0(self_attn)


def tensor_parallelize_model(model):
    decoder_layers = model.model.layers  # ModuleList[ (self_attn, mlp) * 32]
    convert_mlp(decoder_layers)
    return model


def _run_worker():
    dist.init_process_group("nccl")
    args = get_args()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    world_size = dist.get_world_size()
    initialize_parallel_group()
    model = get_model()
    if args.use_tp:
        print_rank_0("====================Use tensor parallel====================")
        model = tensor_parallelize_model(model)

    # return
    model = lora_config_model(model).to("cuda")

    if args.use_amp:
        print_rank_0(
            "====================Use auto mixed precision training===================="
        )

    use_cache = True
    if args.use_grad_ckpt:
        # model.base_model.model.model.gradient_checkpointing = True
        model.base_model.model.gradient_checkpointing_enable()
        use_cache = False
        print_rank_0("====================Use gradient checkpoint====================")

    if args.use_dp:
        # print(   dist.get_group_rank(common.get_val("G_DATA_PARALLEL_GROUP"), dist.get_rank() ) )
        model = DDP(model, process_group=common.get_val("G_DATA_PARALLEL_GROUP"))
        print_rank_0(
            "====================Use distributed data parallel===================="
        )

    # loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, eps=1e-4)
    tokenized_ds = get_dataset()
    ds_6k = tokenized_ds.select(range(6000))
    # ds_6k = tokenized_ds.select(range(64))  # for debug
    # sampler = DistributedSampler(ds_6k, num_replicas=2) if args.use_dp else None

    sampler = (
        DistributedSampler(
            ds_6k,
            rank=dist.get_group_rank(
                common.get_val("G_DATA_PARALLEL_GROUP"), dist.get_rank()
            ),
            num_replicas=2,
        )
        if args.use_dp
        else None
    )
    dl = DataLoader(
        ds_6k,
        batch_size=10,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        sampler=sampler,
    )
    print_rank_0(f"====================get the dataloader====================")
    model.train()
    num_epoch = 1
    log_interval = 1
    scaler = amp.GradScaler()
    for epoch in range(num_epoch):
        print_rank_0(f"====================start train====================")
        if args.use_dp:
            sampler.set_epoch(epoch)
        for batch_idx, data in enumerate(dl):
            if args.use_amp:
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=True
                ):
                    output = model(**data.to(f"cuda:{rank}"), use_cache=use_cache)
                scaler.scale(output.loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(**data.to("cuda"))
                output.loss.backward()
                optimizer.step()
            optimizer.zero_grad()

            if (batch_idx + 1) % log_interval == 0:
                print_rank_0(
                    f"Step: {batch_idx+1}\t Data: {data['input_ids'].shape}\t Training Loss: {output.loss.item()}"
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
    # torchrun --standalone  --nnodes=1 --nproc_per_node=4  train.py --use_dp --use_grad_ckpt --use_amp --use_tp > train.log 2>&1
    # torchrun --standalone  --nnodes=1 --nproc_per_node=2  train.py --use_grad_ckpt --use_amp --use_tp > train.log 2>&1
