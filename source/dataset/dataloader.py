from typing import Optional, List
from omegaconf import DictConfig, open_dict
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.loader import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import StratifiedKFold  
import ipdb
import torch  
import random  

def init_stratified_dataloader(cfg: DictConfig,
                               graph_data_list: List,
                               labels: np.array = None) -> List:
                               
    
    if True:
        kfold = StratifiedKFold(
            n_splits=10,
            shuffle=True,
            random_state=0
        )        
        fold_dataloaders = []
        
        for train_index, test_index in kfold.split(graph_data_list, labels):
            train_data_list = [graph_data_list[i] for i in train_index]
            test_data_list = [graph_data_list[i] for i in test_index]
            
            train_dataloader = DataLoader(
                train_data_list, 
                batch_size=cfg.dataset.batch_size, 
                shuffle=True
            )
            test_dataloader = DataLoader(
                test_data_list,
                batch_size=cfg.dataset.batch_size,
                shuffle=False
            )
            with open_dict(cfg):
                train_length = len(train_data_list)
                cfg.steps_per_epoch = (train_length - 1) // cfg.dataset.batch_size + 1
                cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs
            
            fold_dataloaders.append([train_dataloader, test_dataloader])
        
        return fold_dataloaders
    else:
        train_ratio = cfg.dataset.train_set
        test_ratio = cfg.dataset.test_set

        split = StratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, train_size=train_ratio, random_state=cfg.dataset.seed)
        for train_index, test_index in split.split(graph_data_list, site):
            train_data_list = [graph_data_list[i] for i in train_index]
            test_data_list = [graph_data_list[i] for i in test_index]

        train_dataloader = DataLoader(
            train_data_list, batch_size=cfg.dataset.batch_size, shuffle=True)
        test_dataloader = DataLoader(
            test_data_list, batch_size=cfg.dataset.batch_size, shuffle=False)

        with open_dict(cfg):
            train_length = len(train_data_list)
            cfg.steps_per_epoch = (train_length - 1) // cfg.dataset.batch_size + 1
            cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs
        return [train_dataloader, test_dataloader]