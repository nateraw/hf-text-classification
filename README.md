# hf-text-classification

Text classification using [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), [transformers](https://github.com/huggingface/transformers), and [datasets](https://github.com/huggingface/datasets)

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

  1. Prepare a text classification dataset from the `datasets` package
  2. Train using Lightning's `pl.Trainer` on multiple GPUs with 16 bit precision
  3. Run inference on the dataset's test set and save predictions to file (on multiple GPUs, of course :sunglasses:)

Currently, there are 3 available datasets you can use (pass these to the `--datamodule` flag from CLI):

  - emotion
  - ag_news
  - mrpc

### Examples

#### Fine-tune `bert-base-uncased` for emotion detection

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
