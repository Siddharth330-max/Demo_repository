import sys  
import pandas as pd
from pathlib import Path
from src.exception import CustomException 
from src.utils import load_obj       # load_obj loads pickle files

class PredictPipeline:
    def __init__(self):
        # Set the base path relative to the current file
        self.base_path = Path(__file__).resolve().parent.parent / "components" / "artifacts"

    def predict(self, features):
        try:
            model_path = self.base_path / 'model.pkl'
            preprocessor_path = self.base_path / 'preprocessor.pkl'

            model = load_obj(file_path=str(model_path))
            preprocessor = load_obj(file_path=str(preprocessor_path))
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, SPX, GLD, USO, SLV):
        self.SPX = SPX
        self.GLD = GLD
        self.USO = USO
        self.SLV = SLV 

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "SPX": [self.SPX],
                "GLD": [self.GLD],
                "USO": [self.USO],
                "SLV": [self.SLV],  
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
