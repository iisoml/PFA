import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def build_features(df: pd.DataFrame, target_column: str = "turnaround_time_mins") -> pd.DataFrame:
    """
    Build features for the given DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing raw data.
    target_column (str): The target variable name.

    Returns:
    pd.DataFrame: DataFrame with engineered features.
    """
    
    # Extraction des features temporelles depuis result_time
    df["result_year"]    = df["result_time"].dt.year
    df["result_month"]   = df["result_time"].dt.month
    df["result_day"]     = df["result_time"].dt.day
    df["result_hour"]    = df["result_time"].dt.hour
    df["result_weekday"] = df["result_time"].dt.weekday

    # Catégorisation de l'heure
    def categorize_hour(hour):
        if 6 <= hour < 12:   return "matin"
        elif 12 <= hour < 18: return "apres_midi"
        elif 18 <= hour < 24: return "soir"
        else:                  return "nuit"

    # Catégorisation du jour
    def categorize_weekday(day):
        return "weekend" if day in [5, 6] else "weekday"

    # Catégorisation de la charge de travail
    def categorize_workload(w):
        if w < 10:    return "faible"
        elif w < 30:  return "moyen"
        else:          return "eleve"

    df["time_category"]     = df["result_hour"].apply(categorize_hour)
    df["day_category"]      = df["result_weekday"].apply(categorize_weekday)
    df["workload_category"] = df["lab_workload_last_hour"].apply(categorize_workload)

    # Supprimer les colonnes datetime brutes
    df = df.drop(columns=["result_time", "validation_time"])

    # Reconstitution datetime validation (pour enrichissement de la target)
    df["result_datetime"] = pd.to_datetime(dict(
        year=df["result_year"], month=df["result_month"],
        day=df["result_day"],   hour=df["result_hour"]
    ))

    df["estimated_validation_datetime"] = (
        df["result_datetime"] + pd.to_timedelta(df["turnaround_time_mins"], unit="m")
    )

    df["validation_hour"]     = df["estimated_validation_datetime"].dt.hour
    df["turnaround_category"] = df["validation_hour"].apply(categorize_hour)

    df["target_enriched"] = (
        df["estimated_validation_datetime"].dt.strftime("%d-%m") + " - " +
        df["turnaround_category"]
    )

    # Catégorielles → 'Unknown', numériques → moyenne
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].mean())

    # Drop datetime columns before encoding
    datetime_cols = df.select_dtypes(include="datetime64").columns.tolist()
    df = df.drop(columns=datetime_cols)
    print(f"Dropped datetime columns: {datetime_cols}")

    # One-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=True)
   

   # Sélection des features par importance (Random Forest)

    X_all = df_encoded.drop(columns=["turnaround_time_mins"])
    y_all = df_encoded["turnaround_time_mins"]

    rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_all, y_all)

    importances = pd.Series(rf_selector.feature_importances_, index=X_all.columns)
    

    # Features sélectionnées (issues de l'analyse Random Forest)
    important_features = [
        "admissionweight_73.3",
        "age_Unknown",
        "recent_diagnosis_cardiovascular|ventricular disorders|congestive heart failure",
        "labid",
        "lab_workload_last_hour",
        "result_hour",
        "result_day",
        "labname_fio2",
        "recent_diagnosis_neurologic|seizures|seizures|status epilepticus",
        "labname_total co2",
        "result_weekday",
        "labname_temperature"
    ]

    # Vérifier la disponibilité des features
    missing = [f for f in important_features if f not in df_encoded.columns]
    available = [f for f in important_features if f in df_encoded.columns]

    if missing:
        print("Manquantes :", missing)

    # Utiliser les features disponibles uniquement
    final_features = available

    print("feature engineering completed")

    return df_encoded, final_features