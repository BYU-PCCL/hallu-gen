import os
import torch
from datasets import Dataset
import pandas as pd
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

MODEL_PATH = '../../../models/llama2/hf/Llama-2-7b-chat-hf'
new_model = "LoRA_Model"

####################
# QLoRA parameters #
####################
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

###########################
# bitsandbytes parameters #
###########################
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

################################
# TrainingArguments parameters #
################################
output_dir = "./results"
num_train_epochs = 2
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 100
logging_steps = 1

##################
# SFT parameters #
##################
max_seq_length = None
packing = False
device_map = "auto"

# Load and process the dataset
data_df = pd.read_csv('../factCHD.csv')
data_df['text'] = f"<s>[INST] {data_df['query']} [/INST] {data_df['response']} </s>"
data_df.drop(['query', 'response'], axis=1, inplace=True)
dataset = Dataset.from_pandas(data_df)

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)
        
# Load base model
model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    # quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="wandb"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)