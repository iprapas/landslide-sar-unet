from lit_module import plUNET
from datamodule import BeforeAfterCubeDataModule
import pytorch_lightning as pl
import torch
import argparse
import sys


# a function to read plUNET parameters as program arguments
def add_plUNET_args(parent_parser):
    parser = parent_parser.add_argument_group("plUNET")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--loss", type=str, default='ce')
    return parent_parser


def add_BeforeAfterCubeDataModule_args(parent_parser):
    parser = parent_parser.add_argument_group("BeforeAfterCubeDataModule")
    parser.add_argument("--ds_path", type=str, default='/datacube/hokkaido_japan.zarr')
    parser.add_argument("--ba_vars", type=str, default='vv,vh')
    parser.add_argument("--timestep_length", type=int, default=4)
    # parser.add_argument("--event_start_date", type=str, default='20180905')
    # parser.add_argument("--event_end_date", type=str, default='20180907')
    parser.add_argument("--input_vars", type=str, default='vv_before,vv_after,vh_before,vh_after')
    # parser.add_argument("--target", type=str, default='landslides')
    # parser.add_argument("--train_val_test_split", type=str, default='0.7_0.2_0.1')
    parser.add_argument("--batch_size", type=int, default=64)
    # parser.add_argument("--include_negatives", type=bool, default=False)
    return parent_parser


if __name__ == '__main__':
    # create the top-level parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser = add_plUNET_args(parser)
    parser = add_BeforeAfterCubeDataModule_args(parser)
    # add argument for max_epochs
    parser.add_argument("--max_epochs", type=int, default=10)
    # parse the arguments
    hparams = parser.parse_args()

    # transform ba_vars to list
    # hparams.ba_vars = hparams.ba_vars.split(',')
    # transform input_vars to list
    hparams.input_vars = hparams.input_vars.split(',')
    # transform train_val_test_split to tuple
    # hparams.train_val_test_split = tuple([float(x) for x in hparams.train_val_test_split.split('_')])


    # run the main function
    print(hparams)
    pl.seed_everything(42)
    # create the datamodule from the appropriate parsed arguments
    dm = BeforeAfterCubeDataModule(
        ds_path=hparams.ds_path,
        ba_vars=['vv', 'vh'],
        aggregation='mean',
        timestep_length=hparams.timestep_length,
        event_start_date='20180905',
        event_end_date='20180907',
        input_vars=hparams.input_vars,
        target='landslides',
        train_val_test_split=(0.7, 0.2, 0.1),
        batch_size=hparams.batch_size,
        include_negatives=False
    )

    # create the plUNET from the appropriate parsed arguments
    model = plUNET(
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
        num_channels=len(hparams.input_vars),
        loss=hparams.loss
    )

    # create the trainer with the appropriate parsed arguments
    trainer = pl.Trainer(max_epochs=hparams.max_epochs)

    # train the model
    trainer.fit(model, dm)
