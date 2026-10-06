import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


# ============================================================
# DATA TRANSFORMATION CONFIGURATION
# ============================================================

@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path: str = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


# ============================================================
# DATA TRANSFORMATION CLASS
# ============================================================

class DataTransformation:

    def __init__(self):

        self.data_transformation_config = DataTransformationConfig()


    # ========================================================
    # CREATE PREPROCESSING OBJECT
    # ========================================================

    def get_data_transformer_object(self):

        """
        This function creates the preprocessing pipeline.

        Numerical columns:
            - Missing values are replaced with median
            - StandardScaler is applied

        Categorical columns:
            - Missing values are replaced with most frequent value
            - OneHotEncoder is applied
            - StandardScaler is applied
        """

        try:

            # ------------------------------------------------
            # Numerical columns
            # ------------------------------------------------

            numerical_columns = [
                "writing_score",
                "reading_score"
            ]


            # ------------------------------------------------
            # Categorical columns
            # ------------------------------------------------

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]


            # ------------------------------------------------
            # Numerical Pipeline
            # ------------------------------------------------

            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )


            # ------------------------------------------------
            # Categorical Pipeline
            # ------------------------------------------------

            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler(
                            with_mean=False
                        )
                    )
                ]
            )


            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            logging.info(
                f"Numerical columns: {numerical_columns}"
            )

            logging.info(
                f"Categorical columns: {categorical_columns}"
            )


            # ------------------------------------------------
            # Column Transformer
            # ------------------------------------------------

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    ),
                    (
                        "cat_pipeline",
                        cat_pipeline,
                        categorical_columns
                    )
                ]
            )


            logging.info(
                "Preprocessing object created successfully"
            )


            return preprocessor


        except Exception as e:

            raise CustomException(e, sys)


    # ========================================================
    # INITIATE DATA TRANSFORMATION
    # ========================================================

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:

            # ------------------------------------------------
            # Read train and test data
            # ------------------------------------------------

            logging.info(
                "Reading train and test data"
            )

            train_df = pd.read_csv(train_path)

            test_df = pd.read_csv(test_path)


            logging.info(
                "Train and test data read successfully"
            )


            # ------------------------------------------------
            # Create preprocessing object
            # ------------------------------------------------

            logging.info(
                "Obtaining preprocessing object"
            )

            preprocessing_obj = (
                self.get_data_transformer_object()
            )


            # ------------------------------------------------
            # Target column
            # ------------------------------------------------

            target_column_name = "math_score"


            # ------------------------------------------------
            # Separate input features and target
            # ------------------------------------------------

            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )

            target_feature_train_df = train_df[
                target_column_name
            ]


            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )

            target_feature_test_df = test_df[
                target_column_name
            ]


            logging.info(
                "Input features and target separated successfully"
            )


            # ------------------------------------------------
            # Apply preprocessing
            # ------------------------------------------------

            logging.info(
                "Applying preprocessing object on "
                "training dataframe"
            )

            input_feature_train_arr = (
                preprocessing_obj.fit_transform(
                    input_feature_train_df
                )
            )


            logging.info(
                "Applying preprocessing object on "
                "testing dataframe"
            )

            input_feature_test_arr = (
                preprocessing_obj.transform(
                    input_feature_test_df
                )
            )


            # ------------------------------------------------
            # Combine transformed features + target
            # ------------------------------------------------

            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]


            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]


            logging.info(
                "Train and test arrays created successfully"
            )


            # ------------------------------------------------
            # Save preprocessing object
            # ------------------------------------------------

            logging.info(
                "Saving preprocessing object"
            )

            save_object(
                file_path=(
                    self
                    .data_transformation_config
                    .preprocessor_obj_file_path
                ),
                obj=preprocessing_obj
            )


            logging.info(
                "Preprocessing object saved successfully"
            )


            # ------------------------------------------------
            # Return values
            # ------------------------------------------------

            return (
                train_arr,
                test_arr,
                (
                    self
                    .data_transformation_config
                    .preprocessor_obj_file_path
                )
            )


        except Exception as e:

            raise CustomException(e, sys)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        # ----------------------------------------------------
        # Create DataTransformation object
        # ----------------------------------------------------

        obj = DataTransformation()


        # ----------------------------------------------------
        # Input paths
        # ----------------------------------------------------

        train_path = os.path.join(
            "artifacts",
            "train.csv"
        )

        test_path = os.path.join(
            "artifacts",
            "test.csv"
        )


        # ----------------------------------------------------
        # Run transformation
        # ----------------------------------------------------

        train_arr, test_arr, preprocessor_path = (
            obj.initiate_data_transformation(
                train_path,
                test_path
            )
        )


        # ----------------------------------------------------
        # Success message
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("DATA TRANSFORMATION COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print(
            f"Train data shape : {train_arr.shape}"
        )

        print(
            f"Test data shape  : {test_arr.shape}"
        )

        print(
            f"Preprocessor     : {preprocessor_path}"
        )

        print("=" * 60)
        print()


    except Exception as e:

        print()
        print("=" * 60)
        print("DATA TRANSFORMATION FAILED")
        print("=" * 60)

        print(e)

        print("=" * 60)
        print()