import os
import argparse

import torch
import torch.nn as nn
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

ARG_G = None


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
model = None


def get_model():
    global model
    rank = dist.get_rank()
    model = LlamaForCausalLM.from_pretrained(
        "/home/genghaozhe/workspace/huggingface-models/meta-llama/Llama-2-7b-hf",
        low_cpu_mem_usage=True,
        device_map=device_map[rank],
    )


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


def lora_config_model():
    config = LoraConfig(task_type=TaskType.CAUSAL_LM)
    global model
    model = get_peft_model(model, config)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    # model = model.half()  # 当整个模型都是半精度时，需要将adam_epsilon调大
    model.print_trainable_parameters()
    return model


def print_rank_0(content):
    if dist.get_rank() == 0:
        print(content)


def tensor_parallelize_model():
    pass


def _run_worker():
    dist.init_process_group("nccl")
    # dp_pg = dist.new_group([0, 4])
    # sub_pg = dist.new_subgroups(group_size=2)
    args = get_args()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0 or rank == 1:
        get_model()
        tensor_parallelize_model()
        # create model and move it to GPU with id rank
        # device_id = rank % torch.cuda.device_count()
        print_rank_0(f"Move model from cpu to gpu.")
        model = lora_config_model()
        print_rank_0(f"get the model with lora")
        # model = model.to(device_id) # 26364M 显存
        if args.use_amp:
            print_rank_0("Use auto mixed precision training.")
        if args.use_grad_ckpt:
            print_rank_0("Use gradient checkpoint.")
            model.base_model.model.model.gradient_checkpointing = True

        if args.use_dp:
            print_rank_0("Use distributed data parallel.")
            model = DDP(model, process_group=dist.new_group([0, 1]))

        # loss_fn = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, eps=1e-4)
        tokenized_ds = get_dataset()
        ds_6k = tokenized_ds.select(range(3000))
        # ds_6k = tokenized_ds.select(range(32)) # for debug
        sampler = DistributedSampler(ds_6k, num_replicas=2) if args.use_dp else None
        dl = DataLoader(
            ds_6k,
            batch_size=16,
            collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
            sampler=sampler,
        )
        model.train()
        num_epoch = 1
        log_interval = 10
        scaler = amp.GradScaler()
        for epoch in range(num_epoch):
            if args.use_dp:
                sampler.set_epoch(epoch)
            for batch_idx, data in enumerate(dl):
                if args.use_amp:
                    with torch.autocast(
                        device_type="cuda", dtype=torch.float16, enabled=True
                    ):
                        output = model(**data.to("cuda"))
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
