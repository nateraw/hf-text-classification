# hf-text-classification


## Getting Started

First, I suggest setting up a blank Python 3.7 environment. Then, run the following to clone this repo and install the requirements.

```
git clone https://github.com/nateraw/hf-text-classification.git
cd hf-text-classification
pip install -r requirements.txt
```


## Usage

### Overview

The `train.py` file will:

  1. Prepare the emotion dataset from `datasets` package
  2. Train on 4 GPUs with PyTorch's native 16 bit precision
  3. Run inference on the test set on 4 GPUs w/ native 16 bit precision

### Examples

#### Fine-tune `bert-base-uncased` for emotion detection.

```
python train.py \
    --datamodule emotion \
    --model_name_or_path distilbert-base-uncased \
    --gpus 4 \
    --precision 16 \
    --batch_size 8 \
    --num_workers 16
```


#### Bonus! Prove Nvidia Apex works in place of PyTorch native amp.

You'll need APEX installed on your machine:

```
git clone https://github.com/NVIDIA/apex.git
cd apex/ && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Then, you can simply supply `--amp_backend apex` flag when running from CLI.

```
python train.py \
    --datamodule emotion \
    --gpus 2 \
    --precision 16 \
    --amp_backend apex \
    --limit_train_batches 50 \
    --limit_val_batches 20 \
    --max_epochs 2
```
