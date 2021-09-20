import os
import argparse
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from custom_models import SingleDatasetModel, ModelMode, MultiDatasetModel


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    seed_everything(args.seed)
    if args.model_name == "multitask":
        model = MultiDatasetModel(args)
    else:
        model = SingleDatasetModel(args)
    monitor_metric = "val_loss"
    output_format = "{epoch}-{val_loss:.2f}"
    lr_logger = LearningRateLogger()
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(args.output_dir, output_format),
                                          monitor=monitor_metric,
                                          save_top_k=1,
                                          mode="min",
                                          save_weights_only=True)
    logger = TensorBoardLogger(os.path.join(args.output_dir, "tb"), name="logs")
    trainer = Trainer.from_argparse_args(args, callbacks=[lr_logger],
                                         checkpoint_callback=checkpoint_callback, logger=logger)
    trainer.fit(model)
    # load best checkpoint
    if os.path.exists(checkpoint_callback.best_model_path):
        best_ckpt = torch.load(checkpoint_callback.best_model_path)
        model.load_state_dict(best_ckpt["state_dict"])
    model.set_output_mode(ModelMode.test)
    # run test
    print("Computing stats for test set")
    trainer.test(model, test_dataloaders=model.test_dataloader())
    model.set_output_mode(ModelMode.train_val)
    print("Computing stats for the train+val set")
    trainer.test(model, test_dataloaders=model.train_val_dataloader())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", choices=["single", "multitask"], default="multitask")
    temp_args, _ = parser.parse_known_args()
    model_name = temp_args.model_name
    if model_name == "multitask":
        parser = MultiDatasetModel.add_model_specific_args(parser)
    else:
        parser = SingleDatasetModel.add_model_specific_args(parser)

    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--output_dir", default="./logs")
    parser.add_argument("--tokenizer_path", default="../data/rasmus/cleaned/splits/bmr/tokenizers/bytebpe_combined_30k")
    parser.add_argument("--tokenizer_type", choices=["char", "bpe"], default="bpe")
    parser.add_argument("--context_tokenizer_path", help="Dictionary map from context (subforum) to int",
                        default="contexts")
    parser.add_argument("--max_text_len", type=int, default=64)
    parser.add_argument("--episode_len", type=int, default=5)
    parser.add_argument("--use_time", type=str2bool, default=True)
    parser.add_argument("--use_context", type=str2bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_context", default=True, type=str2bool, help="Train context vectors")
    parser.add_argument("--test_eval_samples", type=int, help="Number of test queries to use for eval",
                        default=1000)
    parser.add_argument("--test_metrics_method", choices=["euclidean", "cosine"], help="Metric used for determining NN",
                        default="cosine")
    parser.add_argument("--save_embeddings", action="store_true", help="Output the embeddings for a run")
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--model_params_text", default="model_type='cnn'|emb_dim=32|final_dim=128")
    parser.add_argument("--model_type_context", choices=["random_init", "pretrained_init"])
    parser.add_argument("--pretrained_context_embedding_path",
                        help="json containing pretrained context embeddings, if any")
    parser.add_argument("--model_params_context", default="emb_dim=32")
    parser.add_argument("--model_params_time", default="emb_dim=32")
    parser.add_argument("--model_params_combined", default="output_dim=32|model_type='ProjectViews'")
    parser.add_argument("--model_params_classwise", default="model_type='sm'")

    args = parser.parse_args()
    main(args)




