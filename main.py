import hydra

import pandas as pd
from omegaconf import DictConfig
from loguru import logger

from preprocessing import Preprocessing


@hydra.main(version_base=None, config_path='.', config_name='config')
def data_preparation(config: DictConfig) -> None:
    if config.preprocessing.active:
        preprocessing = Preprocessing(config)  # init class
        diastolic, systolic = preprocessing()  # call class
        print(diastolic.head())
        print(systolic.head())
    
    if config.translation(config: DictConfig) -> None:
        


if __name__ == '__main__':
    data_preparation()
