import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation


@dataclass
class DataIngestionConfig:

    train_data_path: str = os.path.join(
        "artifacts",
        "train.csv"
    )

    test_data_path: str = os.path.join(
        "artifacts",
        "test.csv"
    )

    raw_data_path: str = os.path.join(
        "artifacts",
        "data.csv"
    )


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info(
            "Entered the data ingestion method or component"
        )

        try:

            # --------------------------------------------------
            # 1. Read dataset
            # --------------------------------------------------

            logging.info("Reading the dataset")

            df = pd.read_csv(
                "notebook/data/stud.csv"
            )

            logging.info(
                "Dataset read successfully"
            )

            # --------------------------------------------------
            # 2. Create artifacts directory
            # --------------------------------------------------

            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )

            # --------------------------------------------------
            # 3. Save raw data
            # --------------------------------------------------

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info(
                "Raw data saved successfully"
            )

            # --------------------------------------------------
            # 4. Train-Test Split
            # --------------------------------------------------

            logging.info(
                "Train test split initiated"
            )

            train_set, test_set = train_test_split(
                df,
                test_size=0.20,
                random_state=42
            )

            # --------------------------------------------------
            # 5. Save train data
            # --------------------------------------------------

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # --------------------------------------------------
            # 6. Save test data
            # --------------------------------------------------

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info(
                "Train and test data saved successfully"
            )

            logging.info(
                "Data ingestion completed successfully"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:

            raise CustomException(e, sys)


# ----------------------------------------------------------
# Main execution
# ----------------------------------------------------------

if __name__ == "__main__":

    try:

        # --------------------------------------------------
        # Data Ingestion
        # --------------------------------------------------

        data_ingestion = DataIngestion()

        train_data, test_data = (
            data_ingestion.initiate_data_ingestion()
        )

        print("\n======================================")
        print("DATA INGESTION COMPLETED")
        print("======================================")

        print("Train data:")
        print(train_data)

        print("\nTest data:")
        print(test_data)

        # --------------------------------------------------
        # Data Transformation
        # --------------------------------------------------

        data_transformation = DataTransformation()

        train_arr, test_arr, preprocessor_path = (
            data_transformation.initiate_data_transformation(
                train_data,
                test_data
            )
        )

        print("\n======================================")
        print("DATA TRANSFORMATION COMPLETED")
        print("======================================")

        print("Train array shape:")
        print(train_arr.shape)

        print("\nTest array shape:")
        print(test_arr.shape)

        print("\nPreprocessor saved at:")
        print(preprocessor_path)

        print("\n======================================")
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("======================================\n")

    except Exception as e:

        print("\n======================================")
        print("PIPELINE FAILED")
        print("======================================")

        print(e)

        print("======================================\n")