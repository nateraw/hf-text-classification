from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')


class TextClassifier(pl.LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            label2id: List[str],
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            weight_decay: float = 0.0,
            warmup_steps: int = 0,
            predictions_file: str = 'predictions.pt',
        ):
        super().__init__()
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(
            self.hparams.model_name_or_path,
            num_labels=len(self.hparams.label2id),
            id2label={v: k for k, v in self.hparams.label2id.items()},
            label2id=self.hparams.label2id
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hparams.model_name_or_path,
            config=self.config
        )
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        del batch['idx']
        self.batch = batch
        outputs = self(**batch)
        loss = outputs[0]
        self.outputs = outputs
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        del batch['idx']
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        self.preds = preds
        self.outputs = outputs
        val_acc = self.accuracy(preds, batch['labels'])
        self.log('val_acc', val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        del batch['labels']
        idxs = batch.pop('idx')
        outputs = self(**batch)
        logits = outputs[0]
        preds = torch.argmax(logits, axis=1)
        self.write_prediction('idxs', idxs, self.hparams.predictions_file)
        self.write_prediction('preds', preds, self.hparams.predictions_file)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer

        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        log_dir = self.trainer.logger.log_dir
        self.hparams.save_dir = str(Path(log_dir) / 'saved_hf_components')
        self.model.save_pretrained(self.hparams.save_dir)
        self.tokenizer.save_pretrained(self.hparams.save_dir)


def preprocess(ds, tokenizer, text_fields, padding='max_length', truncation='only_first', max_length=128):
    ds = ds.map(
        lambda ex: tokenizer(
            ex[text_fields[0]]
            if len(text_fields) < 2
            else list(zip(ex[text_fields[0]], ex[text_fields[1]])),
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        ),
        batched=True,
    )
    ds.rename_column_('label', "labels")
    return ds


def transform_labels(example, idx, label2id: dict):
    str_label = example['labels']
    example['labels'] = label2id[str_label]
    example['idx'] = idx
    return example


class TextClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str = 'bert-base-uncased',
        batch_size: int = 16,
        num_workers: int = 8,
        use_fast: bool = True,
        seed: int = 42
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_fast = use_fast
        self.seed = seed

    def setup(self, stage):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=self.use_fast)
        self.ds = load_dataset(self.dataset_name, self.subset_name)
        self.ds = preprocess(self.ds, tokenizer, text_fields=self.text_fields)
        if self.do_transform_labels:
            self.ds = self.ds.map(transform_labels, with_indices=True, fn_kwargs={'label2id': self.label2id})
        cols_to_keep = [
            x for x in ['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'idx'] if x in self.ds['train'].features
        ]
        self.ds.set_format("torch", columns=cols_to_keep)
        self.tokenizer = tokenizer

    def train_dataloader(self):
        return DataLoader(self.ds['train'], batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.ds['validation'], batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.ds['test'], batch_size=self.batch_size, num_workers=self.num_workers)


class EmotionDataModule(TextClassificationDataModule):
    dataset_name = 'emotion'
    subset_name = None
    text_fields = ['text']
    label2id = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}
    do_transform_labels = True


class MrpcDataModule(TextClassificationDataModule):
    dataset_name = 'glue'
    subset_name = 'mrpc'
    text_fields = ['sentence1', 'sentence2']
    label2id = {"not_equivalent": 0, "equivalent": 1}
    do_transform_labels = False


def parse_args(args=None):
    parser = LightningArgumentParser()
    parser.add_datamodule_args(EmotionDataModule)
    parser.add_model_args(TextClassifier)
    parser.add_trainer_args()
    return parser.parse_lit_args()


if __name__ == '__main__':
    from arguments import LightningArgumentParser
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    dm, model, trainer = main()
    pl.seed_everything(args.datamodule.seed)
    dm = EmotionDataModule.from_argparse_args(args.datamodule)
    dm.setup('fit')
    model = TextClassifier(dm.model_name_or_path, dm.label2id, **vars(args.model))
    model.tokenizer = dm.tokenizer
    model.total_steps = (
        (len(dm.ds['train']) // (args.datamodule.batch_size * max(1, (args.trainer.gpus or 0))))
        // args.trainer.accumulate_grad_batches
        * float(args.trainer.max_epochs)
    )
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)
    # trainer.test(test_dataloaders=test_loader)
