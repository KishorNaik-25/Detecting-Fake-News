import logging
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import CustomLogger
from src.exception import CustomException
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.dataingestion = DataIngestionConfig()
        self.logger = CustomLogger().get_logger()  # Use the CustomLogger

    def initiate_data_ingestion(self):
        try:
            if not os.path.exists('artifacts'):
                os.makedirs('artifacts')

            df = pd.read_csv(r"C:\Users\kishu\PycharmProjects\detecting-fake-news\news\news.csv")
            self.logger.info("DataFrame is uploaded")

            df.to_csv(self.dataingestion.raw_data_path, index = False, header = True)
            print("raw_data_saved",self.dataingestion.raw_data_path)
            self.logger.info("Raw data is saved in artifacts folder")

            train_set, test_set = train_test_split(df, test_size = 0.2,random_state = 42)
            self.logger.info("train and test split is done")

            train_set.to_csv(self.dataingestion.train_data_path, index = False, header = True)
            print("train_data_sved", self.dataingestion.train_data_path)
            self.logger.info("train set of data is saved in artificats folder")

            test_set.to_csv(self.dataingestion.test_data_path, index = False, header = True)
            print("train_data_sved", self.dataingestion.test_data_path)
            self.logger.info("test set of data is saved in artificats folder")

            return(

                self.dataingestion.train_data_path,
                self.dataingestion.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

