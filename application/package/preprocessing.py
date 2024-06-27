import numpy as np
import pandas as pd
from colorama import Fore, Style
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA

from application.package.params import *
from application.package.encoders import transform_time_features, transform_features


def preprocess_features(X: pd.DataFrame) -> np.array:
    def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Scikit-learn pipeline that transforms a cleaned dataset into a preprocessed one

        Stateless operation: "fit transform()" equals "transform()".
        """
        def stateless_standardize(X):
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            return (X - mean) / std

        # Education encoding function
        def encode_education(X):
            edu_mapping = {"Undergraduate": 1, "Graduate": 2, "Postgraduate": 3}
            X = X.apply(lambda x: edu_mapping[x] for x in X)
            return X

        # Relationship encoding function
        def encode_relationship(X):
            rel_mapping = {"Alone": 0, "Partner": 1}
            X = X.apply(lambda x: rel_mapping[x] for x in X)
            return X

        # Time pipe
        time_pipe = make_pipeline(
            FunctionTransformer(transform_time_features),
            FunctionTransformer(stateless_standardize)
        )

        # Education pipe
        edu_pipe = make_pipeline(
            FunctionTransformer(encode_education, validate=False)
        )

        # Relationship pipe
        rel_pipe = make_pipeline(
            FunctionTransformer(encode_relationship, validate=False),
            FunctionTransformer(stateless_standardize, validate=False)
        )

        # Complete pipeline
        pipeline = ColumnTransformer(
            [
                ("time_preproc", time_pipe, ['Dt_Customer', 'Year_Birth']),
                ("edu_preproc", edu_pipe, ['Education']),
                ("rel_preproc", rel_pipe, ['Relationship'])
            ],
            remainder=FunctionTransformer(stateless_standardize, validate=False),  # Apply stateless standardisation to remaining features
            n_jobs=-1
        )

        return pipeline

    print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)

    preprocessor = create_sklearn_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    pca = PCA(n_components=3, random_state=0)
    pca_dp = pd.DataFrame(pca.fit_transform(X_processed), columns=(['feat_1', 'feat_2', 'feat_3']))

    return pca_dp

def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Scikit-learn pipeline that transforms a cleaned dataset into a preprocessed one

        Stateless operation: "fit transform()" equals "transform()".
        """
        def stateless_standardize(X):
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            return (X - mean) / std

        # Education encoding function
        def encode_education(X):
            edu_mapping = {"Undergraduate": 1, "Graduate": 2, "Post graduate": 3}
            return np.array([[edu_mapping[label]] for label in X.flatten()])

        # Relationship encoding function
        def encode_relationship(X):
            rel_mapping = {"Alone": 0, "Partner": 1}
            return np.array([[rel_mapping[label]] for label in X.flatten()])

        # Time pipe
        time_pipe = make_pipeline(
            FunctionTransformer(transform_time_features),
            FunctionTransformer(stateless_standardize)
        )

        # Education pipe
        edu_pipe = make_pipeline(
            FunctionTransformer(encode_education, validate=False)
        )

        # Relationship pipe
        rel_pipe = make_pipeline(
            FunctionTransformer(encode_relationship, validate=False),
            FunctionTransformer(stateless_standardize, validate=False)
        )

        # Complete pipeline
        pipeline = ColumnTransformer(
            [
                ("time_preproc", time_pipe, ['Dt_Customer', 'Year_Birth']),
                ("edu_preproc", edu_pipe, ['Education']),
                ("rel_preproc", rel_pipe, ['Relationship'])
            ],
            remainder=FunctionTransformer(stateless_standardize, validate=False),  # Apply stateless standardisation to remaining features
            n_jobs=-1
        )

        return pipeline
