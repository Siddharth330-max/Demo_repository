import os  
import sys  

import pandas as pd  
import numpy as np 
from dataclasses import dataclass 

from sklearn.preprocessing import StandardScaler  

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj  

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')  

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            target_column_name = "EUR/USD"
            numerical_columns = ['SPX', 'GLD', 'USO', 'SLV']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying Standardisation on training dataframe and testing dataframe.")

            # Create and fit scaler
            scaler = StandardScaler()
            input_feature_train_arr = scaler.fit_transform(input_feature_train_df)
            input_feature_test_arr = scaler.transform(input_feature_test_df)

            # Save scaler
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=scaler
            )

            # Concatenate features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object and transformed arrays.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
