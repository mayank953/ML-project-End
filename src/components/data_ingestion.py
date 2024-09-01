from src.logger import logging
from src.exceptions import CustomException
import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.util import train_test_split_data

from src.components.data_transformation import DataTransformationConfig

from src.components.data_transformation import DataTransformation


from src.components.data_trainer import ModelTrainer

from src.components.data_trainer import ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artefacts','train.csv')
    test_data_path = os.path.join('artefacts','test.csv')
    raw_data_path = os.path.join('artefacts','raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting to read the data")
        try :
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the data as csv")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Splitting the data")
            train_data,test_data = train_test_split_data(df,0.2)

            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion and division completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if(__name__ == "__main__"):
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_tranformation = DataTransformation()

    train_arr,test_arr,_ = data_tranformation.initiate_data_tranformation(train_path,test_path)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr)) 