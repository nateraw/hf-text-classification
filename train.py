import os

import pytorch_lightning as pl

from argument_utils import LightningArgumentParser
from data import EmotionDataModule, MrpcDataModule, AGNewsDataModule
from model import TextClassifier


os.environ["TOKENIZERS_PARALLELISM"] = "true"

datamodule_map = {
    'emotion': EmotionDataModule,
    'mrpc': MrpcDataModule,
    'ag_news': AGNewsDataModule
}

def parse_args(args=None):
    parser = LightningArgumentParser()
    parser.add_argument('--datamodule', type=str)
    temp_args, extras = parser.parse_known_args(args)
    dm_cls = datamodule_map.get(temp_args.datamodule, None)
    if dm_cls is None:
        raise RuntimeError(f'given datamodule: "{temp_args.datamodule}" does not exist')
    parser.add_datamodule_args(dm_cls)
    parser.add_model_args(TextClassifier)
    parser.add_trainer_args()
    return parser.parse_lit_args(extras), dm_cls


if __name__ == '__main__':
    args, dm_cls = parse_args()
    pl.seed_everything(args.datamodule.seed)
    dm = dm_cls.from_argparse_args(args.datamodule)
    dm.setup('fit')
    model = TextClassifier(dm.model_name_or_path, dm.label2id, **vars(args.model))
    model.tokenizer = dm.tokenizer
    model.total_steps = (
        (len(dm.ds['train']) // (args.datamodule.batch_size * max(1, (args.trainer.gpus or 0))))
        // args.trainer.accumulate_grad_batches
        * float(args.trainer.max_epochs)
    )
    trainer = pl.Trainer.from_argparse_args(args.trainer)
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
    model.save_pretrained("outputs")
