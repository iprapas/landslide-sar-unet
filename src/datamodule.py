from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import numpy as np
import xarray as xr
import xbatcher
from collections import defaultdict


def before_after_ds(ds_path, ba_vars, aggregation, timestep_length, event_start_date, event_end_date):
    ds = xr.open_zarr(ds_path)
    for var in ba_vars:
        ds[var] = np.log(ds[var])
    ds = ds.where(ds['sat:orbit_state'] == 'd', drop=True)
    before_ds = ds.drop_dims('timepair').sel(timestep=slice(None, event_start_date))
    after_ds = ds.drop_dims('timepair').sel(timestep=slice(event_end_date, None))

    if timestep_length < len(before_ds['timestep']):
        before_ds = before_ds.isel(timestep=range(-1 - timestep_length, -1))

    if timestep_length < len(after_ds['timestep']):
        after_ds = after_ds.isel(timestep=range(timestep_length))

    if aggregation == 'mean':
        before_ds = before_ds.mean(dim=('timestep'))
        after_ds = after_ds.mean(dim=('timestep'))
    elif aggregation == 'median':
        before_ds = before_ds.median(dim=('timestep'))
        after_ds = after_ds.median(dim=('timestep'))

    before_after_vars = []
    for suffix in ['before', 'after']:
        for var in ba_vars:
            before_after_vars.append(f'{var}_{suffix}')
    the_ds = before_ds.rename_vars({var: f'{var}_before' for var in ba_vars})
    for var in ba_vars:
        the_ds[f'{var}_after'] = after_ds[var]
    for var in the_ds.data_vars:
        the_ds[f'{var}_mean'] = the_ds[var].mean()
        the_ds[f'{var}_std'] = the_ds[var].std()
    return the_ds.load()


def batching_dataset(ds, input_vars, target, include_negatives):
    mean_std_dict = {}
    for var in input_vars:
        if not mean_std_dict.get(var):
            mean_std_dict[var] = {}
        mean_std_dict[var]['mean'] = ds[f'{var}_mean'].values
        mean_std_dict[var]['std'] = ds[f'{var}_std'].values

    batches = []
    bgen = xbatcher.BatchGenerator(ds, {'x': 128, 'y': 128})
    positives = 0
    negatives = 0
    for batch in bgen:
        positives_tmp = batch[target].sum().item()
        if not include_negatives and positives_tmp > 0:
            positives = positives + positives_tmp
            negatives += batch[target].size
            batches.append(batch)
        elif include_negatives and (batch['dem'] <= 0).sum() == 0:
            positives = positives + positives_tmp
            negatives += batch[target].size
            batches.append(batch)
    print(f"P/(P+N)", positives / negatives)
    return batches, mean_std_dict


class BeforeAfterDatasetBatches(Dataset):
    def __init__(self, batches, input_vars, target, mean_std_dict=None):
        print("**************** INIT CALLED ******************")
        self.batches = batches
        self.target = target
        self.input_vars = input_vars
        self.mean = np.stack([mean_std_dict[var]['mean'] for var in input_vars]).reshape((-1, 1, 1))
        self.std = np.stack([mean_std_dict[var]['std'] for var in input_vars]).reshape((-1, 1, 1))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch = self.batches[idx]
        inputs = np.stack([batch[var].values for var in self.input_vars])
        inputs = (inputs - self.mean) / self.std

        target = batch[self.target].values
        inputs = np.nan_to_num(inputs, nan=0)
        target = np.nan_to_num(target, nan=0)
        target = (target > 0)
        return inputs, target


class BeforeAfterCubeDataModule(LightningDataModule):
    """LightningDataModule.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            ds_path: str,
            ba_vars,
            aggregation,
            timestep_length,
            event_start_date,
            event_end_date,
            input_vars,
            target,
            include_negatives=False,
            train_val_test_split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.ds = None
        self.batches = None
        self.mean_std_dict = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.ds = before_after_ds(self.hparams.ds_path, self.hparams.ba_vars, self.hparams.aggregation,
                                      self.hparams.timestep_length, self.hparams.event_start_date,
                                      self.hparams.event_end_date)
            self.batches, self.mean_std_dict = batching_dataset(self.ds, self.hparams.input_vars, self.hparams.target,
                                                                self.hparams.include_negatives)

            dataset = BeforeAfterDatasetBatches(self.batches, self.hparams.input_vars, self.hparams.target,
                                                        mean_std_dict=self.mean_std_dict)
            # self.data_val = BeforeAfterDatasetBatches(self.batches, self.hparams.input_vars, self.hparams.target,
            #                                           mean_std_dict=self.mean_std_dict,
            #                                           )
            # self.data_test = BeforeAfterDatasetBatches(self.batches, self.hparams.input_vars, self.hparams.target,
            #                                            mean_std_dict=self.mean_std_dict,
            #                                            )

            train_val_test_split = [int(len(dataset) * x) for x in self.hparams.train_val_test_split]
            train_val_test_split[2] = len(dataset) - train_val_test_split[1] - train_val_test_split[0]
            train_val_test_split = tuple(train_val_test_split)
            print("*" * 20)
            print("Train - Val - Test SPLIT", train_val_test_split)
            print("*" * 20)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

            print("*" * 20)
            print("Train - Val - Test LENGTHS", len(self.data_train), len(self.data_val), len(self.data_test))
            print("*" * 20)

    def train_dataloader(self):
        return MultiEpochsDataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=(self.hparams.num_workers > 0)
        )

    def val_dataloader(self):
        return MultiEpochsDataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=(self.hparams.num_workers > 0)
        )

    def test_dataloader(self):
        return MultiEpochsDataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=(self.hparams.num_workers > 0)
        )


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
