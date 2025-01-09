import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


from modules.kdd_libs import *
from modules.logger import md_logger




if __name__ == "__main__":

    X_train, y, X_test = load_raw_data()

    X_train = basic_preprocessing(X_train, basic_only=False)
    X_test = basic_preprocessing(X_test, basic_only=False)
    
    log_to = md_logger("results/main_kdd.md")
    log_to.title("KDD #1 - Main - R Chapman")

    log_to.subtitle("Data Preprocessing")
    
    common_stats(X_train, X_test, log_to)

    run_all_default(X_train, y, X_test, log_to)

