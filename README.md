# weightless
How sparse can we make the model?

Read here:
[Project Document](https://docs.google.com/document/d/16mboL_7uz1Hj671S9gyrnwQ2_e6FABsJAbynh_IrNxQ/edit?usp=sharing)

### Your Task
Train a model that achieves val loss less than 3.0 on FineWeb with the fewest number of active parameters. 

## Dataset
This repo uses the tokenized [FineWeb-edu-gpt2](https://huggingface.co/datasets/flappingairplanes/fineweb-edu-gpt2) dataset (GPT-2 tokenizer, 513 token sequences).

## Setup

Using pixi (recommended):
```bash
pixi install
```

Or with pip:
```bash
pip install -r requirements.txt
```

Log in to wandb (optional but recommended):
```bash
wandb login
```

## Files

- `data.py` - Data loading from HuggingFace streaming dataset
- `model.py` - Starter transformer model (modify this!)
- `train.py` - Training loop with wandb logging
- `eval.py` - Evaluation script

## Training

```bash
# Using pixi
pixi run train

# Or directly with python
python train.py

# Custom config
python train.py --batch_size 64 --max_lr 1e-4 --num_steps 20000 --d_model 128 --n_layers 2

# Without wandb
python train.py --no_wandb
```

## Evaluation (not fully implemented)

```bash
# Using pixi
pixi run eval

# Or directly
python eval.py --checkpoint model.pt
```

## Challenge

Your goal: **val_loss < 3.0** with minimal active parameters.

Ideas to explore:
- Smaller architectures
- Weight pruning
- Quantization
- Knowledge distillation
- Novel sparse architectures
- Weight sharing/tying
# sparse_flappy

