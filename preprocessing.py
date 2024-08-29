import os
import hydra
from omegaconf import DictConfig

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

@hydra.main(version_base=None, config_path='.', config_name='config')
class Preprocessing:
    """Creates a longitudinal dataset, by adding repeat_instance columns with a suffix _+ number of repeat instance, 
    and further translating the different arms also to longitudinal format"""
    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def __call__(self):
        self.diastolic_data = pd.read_csv(self.config.preprocessing.redcap_database)
        self.systolic_data = pd.read_csv(self.config.preprocessing.redcap_database)
        
        return self.diastolic_data, self.systolic_data
    

preprocessing = Preprocessing(config)
diastolic_data, systolic_data = preprocessing()

print(diastolic_data.head())
print(systolic_data.head())