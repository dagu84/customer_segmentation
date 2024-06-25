import os

##### VARIABLES #####
DATA_SIZE = os.environ.get("DATA_SIZE")
CHUNK_SIZE = os.environ.get("CHUNK_SIZE")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("EVALUATION_START_DATE")
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
BQ_REGION = os.environ.get("BQ_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL = os.environ.get("MLFLOW_MODEL")
PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
