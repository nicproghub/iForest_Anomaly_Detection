import pandas as pd
import numpy as np
from preprocess_function import detect_peaks

def auc_calculation(org_df, top_peaks_set):
    """
    Calculate the AUC for CO2 and CH4 signals in the given DataFrame.
    """
    df = org_df.copy(deep=True)

    # Initialize columns for new df
    df["is_PeakPy"] = 0
    df["CO2_AUC_t"] = 0.0
    df["CO2_AUC_Sum"] = 0.0
    df["CH4_AUC_Sum"] = 0.0
    df["CH4_AUC_t"] = 0.0
    df["is_CleanPeak"] = 0.0 
    ct  = 0

    # Loop over each session to validate peaks and calculate AUC
    for session_id, session_df in df.groupby('session.ID'):
        # Get the actual indices from the original dataframe
        session_indices = session_df.index
        filtered_df = session_df.copy(deep=True)

        # Remove any NaN CO2 for peak detection and baseline
        valid_co2_mask = filtered_df['CO2_Filter'].notna()
        valid_co2_df = filtered_df[valid_co2_mask].copy()
        
        # If there is data, detect_peaks
        if len(valid_co2_df) > 0:
            ch4_peaks, start_idxs, end_idxs, _ = detect_peaks(filtered_df['CH4'].values,  prominence=0.02, distance=20)

        if len(start_idxs) > 0:
            # Shift start and end for peak period
            co2_start_shifted = start_idxs 
            co2_end_shifted = end_idxs 

            # === AUC calculation logic, loop each peak ===
            for idx, (start, end, peaks) in enumerate(zip(co2_start_shifted, co2_end_shifted, ch4_peaks)):
                # === Mark peak as clean if it's NOT in top_peaks ===
                is_clean = int(((session_id, idx) not in top_peaks_set) and (end > start))
                # Use session_indices instead of filtered_df.index
                peak_idx = session_indices[peaks]
                df.loc[peak_idx, 'is_CleanPeak'] = is_clean

                if is_clean == 0:
                    ct += 1
                    # Skip all AUC calculations for this anomalous peak
                    #continue 

               # AUC calculation
                y = filtered_df['CO2_Filter'].values[start:end+1]
                z = filtered_df['CH4'].values[start:end+1]
                #x = np.arange(s, e+1)
                x = np.arange(end - start + 1)
                # calculate AUC using trapz function interval = peak start - peak end 
                auc_val_sum = np.trapz(y, x)
                auc_ch4_sum = np.trapz(z, x)
                    # loop over each interval [t, t+1]
                for i in range(len(y) - 1): 
                    auc_val = (y[i] + y[i+1]) / 2 * (x[i+1] - x[i])  # trapezoid area for co2 filter
                    df.loc[filtered_df.index[start+i], 'CO2_AUC_t'] = auc_val
                    ch4_auc_val = (z[i] + z[i+1]) / 2 * (x[i+1] - x[i])   # trapezoid area for ch4
                    df.loc[filtered_df.index[start+i], 'CH4_AUC_t'] = ch4_auc_val
                    # baseline_series.iloc[0:end_idxs[0]] = avg_val

                df.loc[peak_idx, 'is_PeakPy'] = 1
                df.loc[peak_idx, 'CO2_AUC_Sum'] = auc_val_sum
                df.loc[peak_idx, 'CH4_AUC_Sum'] = auc_ch4_sum
    
    print(f"Total number of anomalous peaks detected: {ct}")
    
    return df

