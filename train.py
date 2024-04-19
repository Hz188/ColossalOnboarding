import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.checkpoint import checkpoint_sequential

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model

device_map = [
   {
    'model.embed_tokens': 0,
    'model.layers.0': 0,
    'model.layers.1': 0,
    'model.layers.2': 0,
    'model.layers.3': 0,
    'model.layers.4': 0,
    'model.layers.5': 0,
    'model.layers.6': 0,
    'model.layers.7': 0,
    'model.layers.8': 0,
    'model.layers.9': 0,
    'model.layers.10': 0,
    'model.layers.11': 0,
    'model.layers.12': 0,
    'model.layers.13': 0,
    'model.layers.14': 0,
    'model.layers.15': 0,
    'model.layers.16': 1,
    'model.layers.17': 1,
    'model.layers.18': 1,
    'model.layers.19': 1,
    'model.layers.20': 1,
    'model.layers.21': 1,
    'model.layers.22': 1,
    'model.layers.23': 1,
    'model.layers.24': 1,
    'model.layers.25': 1,
    'model.layers.26': 1,
    'model.layers.27': 1,
    'model.layers.28': 1,
    'model.layers.29': 1,
    'model.layers.30': 1,
    'model.layers.31': 1,
    'model.norm': 1,
    'lm_head': 1
   },
   {
    'model.embed_tokens': 2,
    'model.layers.0': 2,
    'model.layers.1': 2,
    'model.layers.2': 2,
    'model.layers.3': 2,
    'model.layers.4': 2,
    'model.layers.5': 2,
    'model.layers.6': 2,
    'model.layers.7': 2,
    'model.layers.8': 2,
    'model.layers.9': 2,
    'model.layers.10': 2,
    'model.layers.11': 2,
    'model.layers.12': 2,
    'model.layers.13': 2,
    'model.layers.14': 2,
    'model.layers.15': 2,
    'model.layers.16': 3,
    'model.layers.17': 3,
    'model.layers.18': 3,
    'model.layers.19': 3,
    'model.layers.20': 3,
    'model.layers.21': 3,
    'model.layers.22': 3,
    'model.layers.23': 3,
    'model.layers.24': 3,
    'model.layers.25': 3,
    'model.layers.26': 3,
    'model.layers.27': 3,
    'model.layers.28': 3,
    'model.layers.29': 3,
    'model.layers.30': 3,
    'model.layers.31': 3,
    'model.norm': 3,
    'lm_head': 3
   }
]

def cleanup():
   dist.destroy_process_group()

tokenizer = AutoTokenizer.from_pretrained("/home/genghaozhe/workspace/huggingface-models/meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = 2
tokenizer.padding_side = "right"  
model = None
def get_model():
   global model
   rank = dist.get_rank()
   model = LlamaForCausalLM.from_pretrained("/home/genghaozhe/workspace/huggingface-models/meta-llama/Llama-2-7b-hf", low_cpu_mem_usage=True, device_map=device_map[rank])

def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ", add_special_tokens=False)
    response = tokenizer(example["output"], add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def get_dataset():
   ds = load_dataset("shibing624/alpaca-zh", split='train')
   tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
   return tokenized_ds

def lora_config_model():
   config = LoraConfig(task_type=TaskType.CAUSAL_LM)
   global model
   model = get_peft_model(model, config)
   model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
   model = model.half()  # 当整个模型都是半精度时，需要将adam_epsilon调大
   model.print_trainable_parameters()
   return model

def print_rank_0(content):
   if dist.get_rank() == 0:
      print(content)

def _run_worker():
   dist.init_process_group("nccl")
   get_model()
   rank = dist.get_rank()
   world_size = dist.get_world_size()
   print_rank_0(f"Start running basic DDP example on rank {rank}, world_size: {world_size}.")

   # create model and move it to GPU with id rank
   # device_id = rank % torch.cuda.device_count()
   print_rank_0(f"Move model from cpu to gpu.")
   model = lora_config_model()
   print_rank_0(f"get the model with lora")
   # model = model.to(device_id) # 26364M 显存
   ddp_model = DDP(model)

   # loss_fn = nn.MSELoss()
   optimizer = optim.AdamW(ddp_model.parameters(), lr=5e-5, eps=1e-4)   
   tokenized_ds = get_dataset()
   ds_6k = tokenized_ds.select(range(6000))
   sampler = DistributedSampler(ds_6k)
   dl = DataLoader(ds_6k,
                   batch_size=16, 
                   collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True), 
                   sampler=sampler)
   # optimizer.zero_grad()
   # outputs = ddp_model(torch.randn(20, 10))
   # labels = torch.randn(20, 5).to(device_id)
   # loss_fn(outputs, labels).backward()
   # optimizer.step()
   num_epoch = 1
   log_interval = 10
   for epoch in range(num_epoch):
      sampler.set_epoch(epoch)
      for batch_idx, data in enumerate(dl):
         optimizer.zero_grad()
         output = ddp_model(**data.to("cuda"))
         output.loss.backward()
         optimizer.step()
         if batch_idx % log_interval == 0:
               print(f"Step: {batch_idx}\t Training Loss: {output.loss.item()}")

   dist.destroy_process_group()


if __name__ == "__main__":
   _run_worker()
   # torchrun --nnodes=1 --nproc_per_node=2 --master_addr=localhost --master_port=12355 train.py