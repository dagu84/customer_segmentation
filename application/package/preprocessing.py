import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from application.package.params import *
from application.package.encoders import transform_time_features, transform_features


def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Scikit-learn pipeline that transforms a cleaned dataset into a preprocessed one

        Stateless operation: "fit transform()" equals "transform()".
        """
        # Time pipe
        time_pipe = FunctionTransformer(transform_time_features)

        # Categorical feature function transformer
        edu_mapping = {"Undergraduate": 1, "Graduate": 2, "Post graduate": 3}
        edu_pipe = FunctionTransformer(np.array(lambda x: [edu_mapping[label] for label in x]).reshape(-1, 1))

        rel_mapping = {"Alone": 0, "Partner": 1}
        rel_pipe = FunctionTransformer(np.array(lambda x: [rel_mapping[label] for label in x]).reshape(-1, 1))

        # Standardisation

        scaler = FunctionTransformer()
