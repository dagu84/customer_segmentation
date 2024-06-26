import pandas as pd
import datetime
from application.package.params import *

def transform_time_features(X: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(X, pd.DataFrame)
    CURRENT_DATE = pd.Timestamp(datetime.date.today())

    # New feature that will find the distance from the most recent timestamp
    X['Customer_For'] = X['Dt_Customer'].apply(lambda x: CURRENT_DATE - x)
    X['Customer_For'] = X['Customer_For'].dt.days.astype(int)

    # Creating a new 'age' feature
    X['Age'] = CURRENT_DATE - X['Year_Birth']

    # Dropping transformed features
    X = X.drop(columns=['Dt_Customer', 'Year_Birth'])

    return X

def transform_features(X: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(X, pd.DataFrame)

    # Aggregating Marital_Status results
    X['Relationship'] = X['Marital_Status'].replace({'Married':'Partner', 'Together':'Partner', 'Absurd':'Alone', 'Widow':'Alone', 'YOLO':'Alone', 'Divorced':'Alone', 'Single':'Alone'})

    # Combining Kidhome and Teenhome into Children
    X['Children'] = X['Kidhome'] + X['Teenhome']

    # Creating a feature that counts the number of people in the household, combining both Relationship and Children
    X['Household'] = X['Relationship'].replace({'Partner': 2, 'Alone': 1}) + X['Children']

    # Creating a Parent feature
    X['Parent'] = X['Children'].apply(lambda x: 1 if x > 0 else 0)

    # Converting the Education results into new easier to interpret categories
    X['Education'] = X['Education'].replace({'Basic':'Undergraduate', '2n Cycle':'Undergraduate', 'Graduation':'Graduate', 'Master':'Postgraduate', 'PhD':'Postgraduate'})

    # Creating a Spent feature, combining Wines, Fruits, Meats, Fish, Sweets and Gold
    X['Spent'] = X['MntWines']+ X['MntFruits']+ X['MntMeatProducts']+ X['MntFishProducts']+ X['MntSweetProducts']+ X['MntGoldProds']

    # Removing uneccessary features
    X = X.drop(columns=['Marital_Status', 'Z_CostContact', 'Z_Revenue', 'ID', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain', 'Response'])

    return X
