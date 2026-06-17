import os
import json
import torch
import numpy as np
import random
import transformers
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    PeftModel, 
    LoraConfig, 
    TaskType, 
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftConfig
)
from trl import SFTTrainer, SFTConfig
import bitsandbytes as bnb
from huggingface_hub.hf_api import HfFolder
import wandb
import multiprocessing

warnings.filterwarnings("ignore")

def optimal_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus*4) if num_gpus else num_cpus - 1
    return optimal_value

NUM_WORKERS = optimal_workers()


def prompt_Questions_Type(data_point):
    system = "You are a Medical Assistant follow the following instruction"

    if len(data_point["context"]) >= 5:
        instruction = f"""
{data_point["instruction"]}

{data_point["context"]}
"""

    else:
        instruction = f"""
{data_point["instruction"]}
"""

    if len(data_point["choices"]) > 0:
        prompt_template ="""{instruction}

{input}

{choices}
"""
        prompt = prompt_template.format(instruction=instruction.strip(),
                                        input=data_point["input"].strip(),
                                        choices=data_point["choices"].strip())
    else:
        prompt_template = """{instruction}

{input}
"""

        prompt = prompt_template.format(instruction=data_point["instruction"].strip(),
                                        input=data_point["input"].strip())

    message = [{"role": "system", "content": system},
               {"role": "user", "content": prompt},
               {"role": "assistant", "content": data_point["output"].strip()}]

    return message


def format_instruct_prompt(data_point, tokenizer, max_seq_length):
    message = prompt_Questions_Type(data_point)
    data_point["prompt"] = tokenizer.apply_chat_template(message,
                                                        tokenize=False,
                                                        max_length = max_seq_length,
                                                        truncation=True)

    if len(tokenizer(data_point["prompt"]).input_ids) > max_seq_length:
        data_point["prompt"] = None

    return data_point

def generate_prompt(model_id, dataset, max_seq_length):
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                            device_map="auto",
                                            trust_remote_code=True
                                            )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    dataset = dataset.map(format_instruct_prompt,
                          num_proc=NUM_WORKERS,
                          fn_kwargs={"tokenizer": tokenizer, "max_seq_length": max_seq_length},
                          desc="Applying chat template", )
    return dataset

@dataclass
class TrainingConfig:
    model_name: str = "./Llama-3.1-8B-Instruct-Medical-Finetune-v4"
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "./Llama-3.1-8B-Instruct-Medical-Finetune-v4"
    max_length: int = 4096

    train_dataset_path: str = "./train_dataset_clean_V2"
    valid_dataset_path: str = "./valid_dataset_clean_V2"

    num_epochs: int = 6
    learning_rate: float = 2.0e-05
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5

    lora_rank: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05

    wandb_project: str = "Llama-3.1-8B-Instruct-Medical-Finetune-v4"
    wandb_api_key: str = "2be7c86a28a2bcbeccdfa66844abfdd19b9bdabf"
    hf_token: str = "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"
    hub_model_id: str = "Llama-3.1-8B-Instruct-Medical-Finetune-v4"

    resume_from_checkpoint: Optional[str] = None


class DeepSpeedConfig:
    @staticmethod
    def get_config(config: TrainingConfig) -> Dict[str, Any]:
        return {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto", 
            "gradient_accumulation_steps": "auto",
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": config.learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.1
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0.0,
                    "warmup_max_lr": config.learning_rate,
                    "warmup_num_steps": config.warmup_steps,
                    "total_num_steps": "auto"
                }
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True, 
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
            },
            "activation_checkpointing": {
                "partition_activations": False,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": False,
            },
            "wall_clock_breakdown": False,
            "memory_breakdown": False,
            "checkpoint": {
                "tag_validation": "ignore",
                "partition_activations": False,
                "use_node_local_storage": False,
                "load_universal": False
            },
            "offload_optimizer": {
                "device": "none"
            }
        }

    @staticmethod
    def save_config(config: TrainingConfig, filepath: str = "ds_config.json"):
        ds_config = DeepSpeedConfig.get_config(config)
        with open(filepath, "w") as f:
            json.dump(ds_config, f, indent=2)
        return filepath


class DatasetProcessor:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def load_and_prepare_dataset(self):
        print("Loading training dataset...")

        train_dataset = Dataset.load_from_disk(self.config.train_dataset_path)

        print(f"Train dataset size: {len(train_dataset)}")

        train_dataset = generate_prompt(
            self.config.base_model_name,
            train_dataset,
            self.config.max_length
        )

        train_dataset = train_dataset.filter(lambda example: example['prompt'] is not None)

        train_dataset = train_dataset.rename_columns({'prompt': 'text'})

        print(f"Final train dataset size: {len(train_dataset)}")
        print(f"Sample text: {train_dataset[0]['text'][:200]}...")

        return train_dataset


class ModelManager:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def set_seed(self, seed: int = 2):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        transformers.set_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def clear_gpu(self):
        torch.clear_autocast_cache()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    def find_all_linear_names(self, model):
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def get_quantization_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=False
        )

    def get_lora_config(self, model):
        target_modules = self.find_all_linear_names(model)
        print(f"Target modules for LoRA: {target_modules}")
        
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none"
        )

    def setup_model_and_tokenizer(self):
        print("Setting up tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left",
            add_eos_token=True,
        )

        print("Loading PEFT configuration...")

        base_model_path = self.config.base_model_name

        print(f"Loading base model: {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=self.get_quantization_config(),
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",  # enable flash attention 2
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id

        print("Pushing tokenizer to hub...")
        print(self.config.hub_model_id)
        tokenizer.push_to_hub(self.config.hub_model_id)

        model.gradient_checkpointing_enable()

        model = prepare_model_for_kbit_training(model)

        lora_config = self.get_lora_config(model)
        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

        model.config.use_cache = False
        model.config.gradient_checkpointing = True

        return model, tokenizer, lora_config


class Llama31Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model_manager = ModelManager(config)
        self.dataset_processor = DatasetProcessor(config)

        self._setup_environment()

    def _setup_environment(self):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

        wandb.login(key=self.config.wandb_api_key)
        if self.config.wandb_project:
            os.environ["WANDB_PROJECT"] = self.config.wandb_project

        os.environ["HF_TOKEN"] = self.config.hf_token
        HfFolder.save_token(self.config.hf_token)

        self.model_manager.set_seed(2)
        if torch.cuda.is_available():
            self.model_manager.clear_gpu()

    def get_training_arguments(self):
        return SFTConfig(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,

            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type="cosine",

            optim="paged_adamw_8bit",
            weight_decay=0.1,
            max_grad_norm=1.0,

            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True,
            fp16=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=4,
            group_by_length=True,

            deepspeed="ds_config.json",

            logging_dir="./logs",
            logging_steps=200,
            save_strategy="steps",
            save_steps=2000,
            save_total_limit=1,

            push_to_hub=True,
            hub_model_id=self.config.hub_model_id,
            hub_strategy="checkpoint",
            hub_token=self.config.hf_token,

            report_to='wandb',
            seed=2,

            dataset_text_field="text",
            packing=True,
            max_length=self.config.max_length,
        )

    def train(self):
        print("="*60)
        print("Llama 3.2 1B Medical Fine-tuning with DeepSpeed & Flash Attention 2")
        print("="*60)
        print(f"Model: {self.config.model_name}")
        print(f"Output: {self.config.output_dir}")
        print(f"Max length: {self.config.max_length}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("="*60)

        print("Preparing DeepSpeed configuration...")
        DeepSpeedConfig.save_config(self.config)

        print("Setting up model and tokenizer...")
        model, tokenizer, lora_config = self.model_manager.setup_model_and_tokenizer()

        print("Preparing dataset...")
        train_dataset = self.dataset_processor.load_and_prepare_dataset()

        os.makedirs(self.config.output_dir, exist_ok=True)

        print("Initializing SFTTrainer...")
        sft_config = self.get_training_arguments()

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            args=sft_config,
            peft_config=lora_config,
        )

        print("Starting training...")
        if self.config.resume_from_checkpoint:
            print(f"Resuming from checkpoint: {self.config.resume_from_checkpoint}")
            trainer_stats = trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)
        else:
            trainer_stats = trainer.train()

        print("Pushing model to hub...")
        trainer.push_to_hub()

        print("Fine-tuning completed successfully!")
        return trainer, trainer_stats


class ModelTester:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def test_model(self):
        print("Loading model for testing...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        peft_config = PeftConfig.from_pretrained(self.config.output_dir)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        model = PeftModel.from_pretrained(base_model, self.config.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.config.output_dir)

        test_prompt = "What are the common symptoms and treatment options for hypertension?"

        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Test Results:")
        print("="*50)
        print(f"Input: {test_prompt}")
        print("-"*50)
        print(f"Response: {response[len(test_prompt):].strip()}")
        print("="*50)


def main():
    config = TrainingConfig()

    config.resume_from_checkpoint = f"{config.output_dir}/checkpoint-192000"

    trainer_manager = Llama31Trainer(config)

    trainer, stats = trainer_manager.train()

    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()