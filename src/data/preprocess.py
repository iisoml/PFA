import pandas as pd
from datetime import datetime, timedelta



def preprocess_data(df: pd.DataFrame, target_column: str = "turnaround_time_mins") -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and encoding categorical variables.

    Parameters:
    df (pd.DataFrame): The input DataFrame to preprocess.

    Returns:
    pd.DataFrame: A preprocessed DataFrame ready for analysis or modeling.
    """
    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace

    reference_date = datetime(2026, 5, 1, 0, 0, 0)

    df["result_time"]     = df["result_time"].apply(lambda x: reference_date + timedelta(minutes=x))
    df["validation_time"] = df["validation_time"].apply(lambda x: reference_date + timedelta(minutes=x))
    df.head()

    # Normalisation des colonnes textuelles

    df["recent_diagnosis"] = df["recent_diagnosis"].str.lower().str.strip()
    df["labname"]          = df["labname"].str.lower().str.strip()



    # Remplacement des valeurs manquantes dans recent_diagnosis
    df["recent_diagnosis"] = df["recent_diagnosis"].fillna("unknown")



    # Corriger l'âge '> 89' → 90
    df["age"] = df["age"].replace({"> 89": 90})
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["admissionweight"] = pd.to_numeric(df["admissionweight"], errors="coerce")



    # Imputation des médianes
    df["age"]             = df["age"].fillna(df["age"].median())
    df["admissionweight"] = df["admissionweight"].fillna(df["admissionweight"].median())


    return df 
