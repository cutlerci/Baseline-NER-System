"""
ChEMU Data Module

Defines a PyTorch Lightning, LightningDataModule for the ChEMU dataset. 

The ChEMU lab series is an annual competition run by Cheminformatics Elsevier Melbourne University lab. The ChEMU 
shared NER (Named Entity Recognition) task seeks to identify chemical compounds along with their roles in a reaction.
Named entity recognition is a sequence classification problem, we seek to tag a sequence of words in a sentence rather than 
classify the sentence in some way. Entities in this dataset are named according to classes such as "reaction step" and 
"reaction product." Our goal is to label words correctly as being members of these classes, in order to glean information 
from sets of patents that are too large for a human to read. 

Authors: Scott, Osman, and Charles
Date: 10/19/2023
"""


import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from data_preprocessing_functions import save_continual_learning_train_dev, save_continual_learning_test
import os

class ChEMUDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, fill_context, create_tensors, continual_step):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.fill_context = fill_context
        self.tensor_dir = "continual_step_tensors"
        self.create_tensors = create_tensors
        self.continual_step = continual_step


    def prepare_data(self):
        # Set create_tensors to false after the tensors have been saved on your machine to avoid preprocessing again
        if self.create_tensors:
            save_continual_learning_train_dev(os.path.join(self.data_dir, "train_conll"), self.tensor_dir, "train_tensor.pth", True)
            save_continual_learning_train_dev(os.path.join(self.data_dir, "train_conll"), self.tensor_dir, "dev_tensor.pth", True)
            save_continual_learning_test(os.path.join(self.data_dir, "train_conll"), self.tensor_dir, "test_tensor.pth", True)

    def setup(self, stage: str):
        # Assign train and validation datasets for use in dataloaders
        if stage == "fit":
            self.chemu_train = torch.load(os.path.join(self.tensor_dir, str(self.continual_step), 'train_tensor.pth'))
            self.chemu_val = torch.load(os.path.join(self.tensor_dir, str(self.continual_step), 'val_tensor.pth'))

        # Assign test dataset for use in dataloader
        if stage == "test":
            self.chemu_test = torch.load(os.path.join(self.tensor_dir, str(self.continual_step), 'test_tensor.pth'))

    def train_dataloader(self):
        return DataLoader(self.chemu_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=8, prefetch_factor=2, drop_last=True,
                          persistent_workers=True, pin_memory=True,)

    def val_dataloader(self):
        return DataLoader(self.chemu_val, batch_size=self.batch_size,
                          num_workers=8, prefetch_factor=2, drop_last=True,
                          persistent_workers=True, pin_memory=True,)

    def test_dataloader(self):
        return DataLoader(self.chemu_test, batch_size=self.batch_size,
                          num_workers=8, prefetch_factor=2, drop_last=True,
                          persistent_workers=True, pin_memory=True,)

    # def predict_dataloader(self):
    #   return DataLoader(self.chemu_predict, batch_size=self.batch_size,
    #                    num_workers=8, prefetch_factor=2, drop_last=True,
    #                   persistent_workers=True, pin_memory=True, )

    #def state_dict(self):
    #    # track whatever you want here
    #    state = {"current_train_batch_index": self.current_train_batch_index}
    #    return state

    #def load_state_dict(self, state_dict):
    #    # restore the state based on what you tracked in (def state_dict)
    #    self.current_train_batch_index = state_dict["current_train_batch_index"]