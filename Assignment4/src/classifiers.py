import os
from sklearn.tree import DecisionTreeClassifier
from abc import ABC, abstractmethod
from src import Preprocessor

class Classifier(ABC):
    def __init__(self, preprocessor: Preprocessor, class_column: str):
        self._preprocessor = preprocessor
        self._class_column = class_column
        self._model = None
    
    @property
    def model(self):
        return self._model
    
class DecisionTree(Classifier):
    def __init__(self, preprocessor):
        super().__init__(preprocessor)
        self._model = DecisionTreeClassifier()
