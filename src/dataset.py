
import pandas as pd

import logging

import src.config.constants as c
import src.config.settings as s

logger = logging.getLogger(__name__)

class EnergyDataset:
      def __init__(self):
            self.data_path = ""
      
      def _load_data(self) :
            try:
                  self._data = pd.read_csv(self.data_path)
                  logger.info("Energy Consumption Dataset loaded.")
            except:
                  logger.error("unable to load Energy Consumption Dataset.")
      
      def __getitem__(self, idx):
            return self.data.iloc[idx]
      
      def __len__(self):
            return self._data.shape[0]