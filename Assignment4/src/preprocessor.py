import pandas as pd
import os, re
from typing import Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, LabelEncoder

PROCESSED_DATA_PATH = "./processed_data/"

class Preprocessor:
    def __init__(self):
        # keep track of columns already being transformed
        self._column_map = {}
        
        # operations
        self._transformers = {
            "ohe": {
                "transformer": OneHotEncoder(drop="if_binary", sparse_output=False),
                "columns": []
            },
            "oe": {
                "transformer": OrdinalEncoder(),
                "columns": []
            },
            "nmm": {
                "transformer": MinMaxScaler(),
                "columns": []
            }
        }
        self._label_column: str = ""
        self._to_fill_empty: dict = {}
        self._to_drop_columns: list = []
        
        # The "training" data is all of the labeled data
        self._train_save_path: str = ""
        self._train_data: Optional[pd.DataFrame] = None
        
        # The "test predictions" data is the data on which the final predictions should be done
        self._test_predictions_save_path: str = ""
        self._test_predictions_data: Optional[pd.DataFrame] = None
        
        self._column_tranformer: Optional[ColumnTransformer] = None
        self._label_encoder: LabelEncoder = LabelEncoder()
        
    def _set_col(self, name, abrev, columns):
        for column in columns:
            if column in self._column_map:
                print(f"ERROR: [PRE-PROCESS] Can not set columns: '{column}' to {name}, it is already mapped to {self._column_map[column]}")
                continue
            print(f"[PRE-PROCESS] Set column: '{column}' to {name}")
            self._transformers[abrev]["columns"].append(column)
            self._column_map[column] = name
            
    def _perform_fill(self, df):
        for col, data in self._to_fill_empty.items():
            if data["condition"] is None:
                df[col] = df[col].fillna(data["value"])
            else:
                df[col] = df.apply(data["condition"], axis=1)
    
    def OneHotEncode(self, columns):
        self._set_col(name="OneHotEncode", abrev="ohe", columns=columns)
    
    def OrdinalEncode(self, columns):
        self._set_col(name="OrdinalEncode", abrev="oe", columns=columns)
        
    def DropColumns(self, columns):
        print(f"[PRE-PROCESS] Set columns: {columns} to be dropped")
        self._to_drop_columns += columns
    
    def LabelEncode(self, column: str):
        print(f"[PRE-PROCESS] Set column: '{column}' as the target label")
        self._label_column = column
        
    def Normalize(self, columns):
        self._set_col(name="NormalizeMinMax", abrev="nmm", columns=columns)
    
    def FillEmpty(self, column: str, value: Optional[any]=None, condition: Optional[callable]=None):
        if value is None and condition is None:
            print("ERROR: [PRE-PROCESS] FillEmpty, value and condition can not both be empty")
        print(f"[PRE-PROCESS] Set column: '{column}' to fill empty slots with {value if value is not None else 'a condition'}")
        self._to_fill_empty[column] = {
            "value": value,
            "condition": condition
        }
            
    def process(self, train_data_path, test_prediction_data_path, save_to_file=False, remainder="passthrough"):
        self._train_data = pd.read_csv(train_data_path)
        self._test_predictions_data = pd.read_csv(test_prediction_data_path)
        
        # Encode the target label (that are only in the training data)
        if not self._label_column or self._label_column not in self._train_data:
            print("ERROR: [PRE-PROCESS] label column not inside training data, or label not specified")
            return
        raw_label_col = self._train_data.pop(self._label_column)
        encoded_label_col = self._label_encoder.fit_transform(raw_label_col)
        
        # Fill empty data
        self._perform_fill(self._train_data)
        self._perform_fill(self._test_predictions_data)
        
        # Drop columns
        if len(self._to_drop_columns) > 0:
            self._train_data = self._train_data.drop(columns=self._to_drop_columns, errors="ignore")
            self._test_predictions_data = self._test_predictions_data.drop(columns=self._to_drop_columns, errors="ignore")
        
        # Create (name, transformer, columns) tuples
        tuples = []
        for name, data in self._transformers.items():
            if len(data["columns"]) == 0:
                continue
            tuples.append((name, data["transformer"], data["columns"]))
        
        # Apply transformers
        print("[PRE-PROCESS] Applying preprocessings")
        self._column_transformer = ColumnTransformer(tuples, remainder=remainder, verbose_feature_names_out=False)
        train_data_numpy = self._column_transformer.fit_transform(self._train_data) # Fit and transform the train data
        test_data_numpy = self._column_transformer.transform(self._test_predictions_data) # Only transform the test data so there is no mismatch between train data transformations
        
        ## Turn into DataFrame
        feature_names = self._column_transformer.get_feature_names_out()
        self._train_data = pd.DataFrame(train_data_numpy, columns=feature_names)
        self._test_predictions_data = pd.DataFrame(test_data_numpy, columns=feature_names)
        
        # Re-attach encoded label column to train data
        self._train_data[self._label_column] = encoded_label_col
            
        # Turn all column names to snake case
        # https://stackoverflow.com/questions/74643621/convert-dataframe-column-names-from-camel-case-to-snake-case
        self._train_data.columns = self._train_data.columns.str.replace(r'[ ,]+', '_', regex=True).str.lower()  # Snake case all the columns
        self._test_predictions_data.columns = self._test_predictions_data.columns.str.replace(r'[ ,]+', '_', regex=True).str.lower()   # Snake case all the columns
        
        # Save processed data to file
        if save_to_file:
            self._train_save_path = os.path.join(PROCESSED_DATA_PATH, os.path.basename(train_data_path).split('.')[0] + "_processed.csv")
            self._test_predictions_save_path = os.path.join(PROCESSED_DATA_PATH, os.path.basename(test_prediction_data_path).split('.')[0] + "_processed.csv")
        
            self._train_data.to_csv(self._train_save_path)
            self._test_predictions_data.to_csv(self._test_predictions_save_path)
    
    @property
    def train_data(self) -> Optional[pd.DataFrame]:
        if self._train_data is None:
            print("[PRE-PROCESS] train_data is None, call 'process()' first")
        return self._train_data
    @property
    def test_predictions_data(self) -> Optional[pd.DataFrame]:
        if self._test_predictions_data is None:
            print("[PRE-PROCESS] test_data is None, call 'process()' first")
        return self._test_predictions_data
    
    @property
    def column_transformer(self) -> Optional[ColumnTransformer]:
        if self._column_tranformer is None:
            print("[PRE-PROCESS] column_transformer is None, call 'process()' first")
        return self._column_tranformer
            
    