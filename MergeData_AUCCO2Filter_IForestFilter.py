import sys
import os
from datetime import datetime as dt
import pandas as pd
import numpy as np

# Add modules to path
current_dir = os.path.dirname(__file__)
modules_path = os.path.join(current_dir, 'iForest_modules')
sys.path.append(modules_path)

from preprocess_function import (
    detect_peaks, 
    create_peak_features, 
    preprocess_data)
from iforest_model import (
    train_isolation_forest, 
    get_anomalous_peaks)
from auc_calculation import auc_calculation

def main():
    # Configuration
    #
    # ---------------------------------------------------------------
    # -----------------------                 -----------------------
    # -------------------                         ------------------- 
    # -------** Change anomaly_threshold & Source File Here **-------
    # -------------------                         ------------------- 
    # -----------------------                 -----------------------
    # ---------------------------------------------------------------    
    #
    anomaly_threshold = 0.2  # threshold for anomaly score to filter peaks
    # today's date in yyyymmdd format
    today_date = dt.today().strftime("%Y%m%d")

    # File paths
    file_path = r"\Methane Data\MergedData.csv"
    org_df = pd.read_csv(file_path, parse_dates=['Timestamps'])
    df1 = org_df.copy()
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = preprocess_data(org_df)
    
    # Create peaks and features
    print("Creating peaks and features...")
    peaks_all = create_peak_features(df)

    # Train Isolation Forest model
    print("Training Isolation Forest model...")
    peak_features, importance_df = train_isolation_forest(peaks_all)
    fn = f"peak_features_with_anomalies_{today_date}.csv"
    peak_features.to_csv(fn, index=False)
    importance_df.to_csv(f"feature_importance_{today_date}.csv", index=False) 
    print(f"\nResults saved to: {fn}")

    # Get anomalous peaks based on threshold
    print("Identifying anomalous peaks based on threshold...")
    top_peaks, top_peaks_set= get_anomalous_peaks(peak_features, anomaly_threshold=anomaly_threshold)
    print(f"Total anomalous peaks identified: {len(top_peaks)}")      
    fn_anom = f"anomalous_peaks_{today_date}.csv"
    top_peaks.to_csv(fn_anom, index=False)

    # Calculate AUC and mark clean peaks
    print("Calculating AUC and marking clean peaks...")
    new_df = auc_calculation(df, top_peaks_set)


    df1["is_PeakPy"] = new_df["is_PeakPy"]
    df1["CO2_AUC_t"] = new_df["CO2_AUC_t"]
    df1["CO2_AUC_Sum"] = new_df["CO2_AUC_Sum"]
    df1["CH4_AUC_t"] = new_df["CH4_AUC_t"]
    df1["CH4_AUC_Sum"] = new_df["CH4_AUC_Sum"]
    df1["is_CleanPeak"] = new_df["is_CleanPeak"]


    # extract filename without extension 
    print("Saving final results...")
    base_name = file_path.split("\\")[-1].replace(".csv", "")
    # save CSV with date in filename
    df1.to_csv(f"{base_name}_AUC_CO2Filter{today_date}_iForestFilter.csv", index=False)


if __name__ == "__main__":
    main()