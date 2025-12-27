# Llama 3.2 Medical Fine-tuning

This repository contains code for fine-tuning Llama 3.2 models (1B, 3B, and 8B) on medical datasets using LoRA (Low-Rank Adaptation), 4-bit quantization, and DeepSpeed for efficient multi-GPU training.

## Features

- **Multiple Model Sizes**: Training scripts for 1B, 3B, and 8B parameter models
- **4-bit Quantization**: Memory-efficient training using BitsAndBytes
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning with rank 128
- **DeepSpeed ZeRO Stage 2**: Distributed training optimization
- **Flash Attention 2**: Fast and memory-efficient attention mechanism
- **Gradient Checkpointing**: Reduced memory footprint
- **Mixed Precision Training**: BF16 for faster training
- **WandB Integration**: Training monitoring and logging
- **Hugging Face Hub**: Automatic model pushing
- **Comprehensive Evaluation**: LM-Eval, ROUGE, BLEU metrics and memory analysis

## Repository Structure

```
.
├── trainings/
│   ├── main_1b.py          # Training script for 1B model
│   ├── main_3b.py          # Training script for 3B model
│   └── main_8b.py          # Training script for 8B model
├── dataset/
│   └── llama3_finetuning_Dataset.ipynb  # Dataset collection, cleaning & preprocessing
├── Evaluation/
│   ├── Lm_evaluate.ipynb              # LM-Eval harness evaluation
│   ├── rouge-blue-evaluate.ipynb      # ROUGE & BLEU metrics evaluation
│   └── memory_analysis/               # Memory consumption analysis
└── README.md
```

## Requirements

### System Requirements
- Python 3.8+
- CUDA 11.8+ or 12.1+
- 2x GPU with at least 24GB VRAM each (recommended)
- 64GB+ RAM (recommended)

### Installation

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install DeepSpeed
pip install deepspeed

# Install other dependencies
pip install transformers datasets accelerate peft trl bitsandbytes
pip install flash-attn --no-build-isolation
pip install wandb huggingface_hub
```

## Configuration

### Choose Your Model Size

Navigate to the `trainings/` directory and select the appropriate training script:

- **1B Model**: `main_1b.py` - Fastest training, lowest memory requirements
- **3B Model**: `main_3b.py` - Balanced performance and resource usage
- **8B Model**: `main_8b.py` - Best performance, highest memory requirements

### Update Training Configuration

Edit the `TrainingConfig` class in your selected training script (e.g., `main_1b.py`):

```python
@dataclass
class TrainingConfig:
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    output_dir: str = "./Llama-3.2-1B-Instruct-Medical-Finetune-v5"
    
    # Dataset paths
    train_dataset_path: str = "./train_dataset_clean_V2"
    valid_dataset_path: str = "./valid_dataset_clean_V2"
    
    # Training hyperparameters
    num_epochs: int = 6
    learning_rate: float = 2.0e-05
    per_device_batch_size: int = 10
    gradient_accumulation_steps: int = 40
    
    # LoRA parameters
    lora_rank: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    
    # API Keys (replace with your own)
    wandb_api_key: str = "YOUR_WANDB_API_KEY"
    hf_token: str = "YOUR_HF_TOKEN"
    hub_model_id: str = "YOUR_USERNAME/model-name"
```

### Dataset Format

The `dataset/llama3_finetuning_Dataset.ipynb` notebook contains the complete pipeline for:
- **Data Collection**: Gathering medical datasets from various sources
- **Data Cleaning**: Removing duplicates, handling missing values, filtering quality
- **Preprocessing**: Formatting data for fine-tuning with proper prompt templates
- **Validation**: Ensuring data quality and consistency

Your processed dataset should have the following columns:
- `instruction`: The task instruction
- `input`: The input text/question
- `output`: The expected response
- `context`: Additional context (optional)
- `choices`: Multiple choice options (optional)

## Dataset Preparation

Before training, prepare your dataset using the provided notebook:

```bash
# Navigate to dataset directory
cd dataset/

# Open and run the preprocessing notebook
jupyter notebook llama3_finetuning_Dataset.ipynb
```

This will generate the cleaned datasets required for training:
- `train_dataset_clean_V2/`
- `valid_dataset_clean_V2/`

## Running Training

### Select Your Model

Choose the appropriate training script from the `trainings/` directory based on your requirements:

| Model | Script | GPU Memory (per GPU) | Training Speed | Best For |
|-------|--------|---------------------|----------------|----------|
| 1B | `main_1b.py` | ~20-22GB | Fastest | Quick experiments, resource-constrained |
| 3B | `main_3b.py` | ~30-35GB | Medium | Balanced performance |
| 8B | `main_8b.py` | ~40-45GB | Slower | Best accuracy |

### Option 1: DeepSpeed with 2 GPUs (Recommended)

```bash
# Navigate to trainings directory
cd trainings/

# For 1B model
deepspeed --num_gpus=2 main_1b.py

# For 3B model
deepspeed --num_gpus=2 main_3b.py

# For 8B model
deepspeed --num_gpus=2 main_8b.py
```

### Option 2: DeepSpeed with Specific GPUs

```bash
# For 1B model on GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 main_1b.py

# For 3B model
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 main_3b.py

# For 8B model
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 main_8b.py
```

### Option 3: Single GPU (No DeepSpeed)

Comment out the deepspeed line in training arguments and run:

```bash
cd trainings/
python main_1b.py  # or main_3b.py, main_8b.py
```

### Option 4: Using Accelerate

```bash
cd trainings/
accelerate launch --num_processes=2 --mixed_precision bf16 main_1b.py
```

## Model Evaluation

After training, evaluate your fine-tuned model using the notebooks in the `Evaluation/` directory:

### 1. LM-Eval Harness Evaluation

Comprehensive evaluation using the Language Model Evaluation Harness:

```bash
cd Evaluation/
jupyter notebook Lm_evaluate.ipynb
```

**Metrics evaluated:**
- Medical question answering accuracy
- Domain-specific benchmarks (MedQA, PubMedQA, etc.)
- General language understanding tasks
- Few-shot learning performance

### 2. ROUGE & BLEU Metrics

Traditional NLG metrics for response quality:

```bash
cd Evaluation/
jupyter notebook rouge-blue-evaluate.ipynb
```

**Metrics evaluated:**
- **ROUGE-1, ROUGE-2, ROUGE-L**: N-gram overlap and longest common subsequence
- **BLEU**: Precision-based metric for text generation quality
- **Comparison**: Before vs. after fine-tuning

### 3. Memory Consumption Analysis

Analyze GPU memory usage during inference:

```bash
cd Evaluation/
# Memory analysis results are generated in the evaluation notebooks
```

**Analysis includes:**
- Peak memory usage per model size
- Memory vs. sequence length
- Batch size impact
- Quantization effectiveness
- Inference optimization recommendations

## Resume Training from Checkpoint

To resume training from a checkpoint, uncomment and modify this line in `main()`:

```python
config.resume_from_checkpoint = f"{config.output_dir}/checkpoint-2500"
```

Then run the training command again.

## DeepSpeed Configuration

The script automatically generates a `ds_config.json` file with the following features:

- **ZeRO Stage 2**: Optimizer state partitioning
- **BF16 Training**: Mixed precision for faster training
- **Gradient Accumulation**: Automatic handling
- **AdamW Optimizer**: With weight decay
- **Warmup Scheduler**: Linear warmup with decay

You can customize the DeepSpeed configuration in the `DeepSpeedConfig` class.

## Training Output

The training process will:
1. Load and preprocess datasets
2. Apply chat template formatting
3. Initialize model with 4-bit quantization
4. Apply LoRA adapters
5. Train the model
6. Save checkpoints every 500 steps
7. Push final model to Hugging Face Hub
8. Log metrics to WandB

### Output Structure

```
Llama-3.2-1B-Instruct-Medical-Finetune-v5/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-2500/
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer files
```

## Memory Requirements

With the default configuration:

| Model | Per GPU Memory | Effective Batch Size | Max Seq Length |
|-------|----------------|---------------------|----------------|
| 1B | ~20-22GB | 800 (10×40×2) | 1024 tokens |
| 3B | ~30-35GB | 640 (8×40×2) | 1024 tokens |
| 8B | ~40-45GB | 480 (6×40×2) | 1024 tokens |

To reduce memory usage:
- Decrease `per_device_batch_size`
- Decrease `max_length`
- Decrease `lora_rank`

## Testing the Model

Uncomment the testing code in the `main()` function of your training script:

```python
print("\nTesting the fine-tuned model...")
tester = ModelTester(config)
tester.test_model()
```

For comprehensive evaluation, use the notebooks in the `Evaluation/` directory.

## Monitoring Training

### WandB Dashboard
- Training loss
- Learning rate schedule
- GPU memory usage
- Training throughput

### Hugging Face Hub
- Automatic checkpoint uploads
- Model versioning
- Easy deployment

## Troubleshooting

### Out of Memory Error
1. Reduce `per_device_batch_size`
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_length`
4. Enable CPU offloading in DeepSpeed config

### DeepSpeed Import Error
```bash
pip install deepspeed --no-cache-dir
```

### Flash Attention Error
```bash
pip install flash-attn --no-build-isolation
```

### CUDA Version Mismatch
Ensure PyTorch CUDA version matches your system CUDA:
```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

## Performance Tips

1. **Use NVMe SSD**: For faster dataset loading
2. **Pin Memory**: Already enabled in the config
3. **Group by Length**: Enabled to reduce padding overhead
4. **Packing**: Enabled for efficient sequence packing
5. **Multiple Workers**: Automatically configured based on CPU count
6. **Model Selection**: Start with 1B for experimentation, scale to 3B/8B for production

## Quick Start Guide

```bash
# 1. Clone the repository
git clone <repository-url>
cd <repository-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare dataset
cd dataset/
jupyter notebook llama3_finetuning_Dataset.ipynb
# Run all cells to generate cleaned datasets

# 4. Train model (choose your size)
cd ../trainings/
deepspeed --num_gpus=2 main_1b.py  # For 1B model

# 5. Evaluate model
cd ../Evaluation/
jupyter notebook Lm_evaluate.ipynb
jupyter notebook rouge-blue-evaluate.ipynb
```

## Citation

If you use this code, please cite:

```bibtex
@software{llama32_medical_finetune,
  title={Llama 3.2 1B Medical Fine-tuning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/repo}
}
```

## License

This project is licensed under the same license as the base Llama 3.2 model. Please refer to Meta's Llama license for details.

## Acknowledgments

- Meta AI for Llama 3.2
- Hugging Face for Transformers and PEFT
- Microsoft for DeepSpeed
- The open-source community