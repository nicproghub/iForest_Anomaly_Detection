import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, pearsonr
from scipy.signal import find_peaks, peak_widths, peak_prominences
#---------------------------------------------------
# --------- Define Peak Function     ---------------
# --------------------------------------------------

def detect_peaks(signal, prominence=0.02, distance=20,  low_threshold = 0.03, percent = 0.003, n=1 ):
    peaks, _ = find_peaks(signal, prominence=prominence, distance=distance)
    #results_half = peak_widths(signal, peaks, rel_height=0.9)
    #widths, start_idx, end_idx = results_half[0], results_half[2], results_half[3]
    prominences, left_bases, right_bases = peak_prominences(signal, peaks)
    start_idx = left_bases.copy()
    end_idx = right_bases.copy()
    if len(peaks) == 0:
            return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    tot = len(signal)
    low_threshold = 0.03 #* peak_height  # 10% of peak height
    percent = 0.003
    flat_slope=0.01
    flat_points=20

    for i, peak in enumerate(peaks):
        # --- Adjust overlaps between consecutive peaks ---
        if i < len(peaks) - 1:
            if end_idx[i] >= start_idx[i+1] and end_idx[i] >= end_idx[i+1]:
                end_idx[i] = start_idx[i+1]-2
                if start_idx[i+1]-2 <= peaks[i]: # if new start ind < previous peak ind
                    end_idx[i] = peaks[i] + (peaks[i+1] -peaks[i])/2
                    start_idx[i+1] = end_idx[i]+1
            elif end_idx[i] >= start_idx[i+1]:
                start_idx[i+1] = end_idx[i]+2

        # --- Refine start ---
        for j in range(peak, start_idx[i], -1):
            # Clip indices to avoid out-of-bounds
            start1 = max(j - flat_points, 0)
            start2 = max(start1 - n, 0)
            segment = signal[start1:j+1]
            seg2 = signal[start2:j+1-n]
            # Make sure segment lengths match
            min_len = min(len(segment), len(seg2))
            segment = segment[-min_len:]
            segment = segment[-min_len:]
            seg2 = seg2[-min_len:]

            if len(segment) == 0:
                break
            #segment_diffs = np.abs(np.diff(segment))
            # Use difference between current point and point flat_points before
            flat_ratio = np.sum(np.abs(segment - seg2)) / len(segment)
            if flat_ratio < percent and signal[j] < low_threshold:
                start_idx[i] = j
                break

        # --- Refine end ---
        for j in range(peak, end_idx[i]):
            end1 = min(j + flat_points + 1, tot)
            end2 = min(j + flat_points + 1 + n, tot)
            segment = signal[j:end1]
            seg2 = signal[j + n:end2]
            min_len = min(len(segment), len(seg2))
            segment = segment[:min_len]
            seg2 = seg2[:min_len]

            if len(segment) == 0:
                break

            flat_ratio = np.sum(np.abs(segment - seg2)) / len(segment)
            if flat_ratio < percent and signal[j] < low_threshold:
                end_idx[i] = j
                break

    return peaks, start_idx.astype(int), end_idx.astype(int), prominences



#---------------------------------------------------
# --------- peak features extraction ---------------
# --------------------------------------------------

def create_peak_features(df):
    """
    Extracts peak-based features at the session level.

    Parameters
    ----------
    df : pd.DataFrame
        Must include columns: ['session_number', 'time', 'ch4_reading', 'co2_reading', 'animal_number', 'device_number', 'lactation_day']
    prominence, distance : float, int
        Parameters for detect_peaks()

    Returns
    -------
    peak_features_df : pd.DataFrame
        One row per peak with extracted features.
    """

    peak_feature_list = []

    for session_id, session_data in df.groupby('session.ID'):
        ch4 = session_data['CH4'].values
        co2 = session_data['CO2_Filter'].values
        t = session_data['Timestamps'].values

        # Detect peaks on CH4
        peaks, start_idx, end_idx, pro = detect_peaks(ch4,  prominence=0.02, distance=20)

        # Session length
        session_length = len(session_data)

        if len(peaks) > 0:
            #if session_id == 2326:
            #    print(f"Session {session_id} - Peaks detected: {len(peaks)}")
            # Statistical summaries across peaks
            num_peaks = len(peaks)
            avg_peak_height = np.mean(ch4[peaks]) if len(peaks) > 0 else 0

            for i, peak in enumerate(peaks):
                # Peak length
                peak_length = end_idx[i] - start_idx[i]
                if peak_length <= 0:
                    continue
                
                peak_segment = ch4[start_idx[i]:end_idx[i]+1]
                co2_segment = co2[start_idx[i]:end_idx[i]+1]
                # Peak-level features
                peak_ratio = peak_length / session_length if session_length > 0 else 0
                peak_height = ch4[peak]
                peak_std = np.std(peak_segment, ddof=0) if len(peak_segment) > 0 else 0
                # transform peak_std to inversely related feature (lower std -> higher value)
                peak_flat_std = np.log1p(1/(peak_std + 1e-9))

                # Handle skewness and kurtosis with proper conditions
                if len(peak_segment) > 2 and not np.all(peak_segment == peak_segment[0]):
                    peak_skewness = skew(peak_segment)
                    peak_kurtosis = kurtosis(peak_segment)
                else:
                    peak_skewness = 0
                    peak_kurtosis = 0
                    
                # calculate the diff of ch4 at start and end of peak
                s_e_diff = np.abs(ch4[start_idx[i]] - ch4[end_idx[i]]) if len(peak_segment) > 0 else 0
                left_count = peaks[i] - start_idx[i]
                right_count = end_idx[i] - peaks[i]
                left_right_ratio = left_count / right_count * s_e_diff if right_count > 0 else np.nan
                right_left_ratio = right_count / left_count  * s_e_diff if left_count > 0 else np.nan

                # Relationship features within peak
                if len(peak_segment) > 1 and len(co2_segment) > 1 and np.std(peak_segment) > 1e-8 and np.std(co2_segment) > 1e-8:
                    corr = pearsonr(peak_segment, co2_segment)[0]
                else:
                    corr = 0
                ratio = peak_segment / np.where(co2_segment == 0, np.nan, co2_segment)
                ratio_mean = np.nanmean(ratio) if len(ratio) > 0 and not np.all(np.isnan(ratio)) else 0
                ratio_std = np.nanstd(ratio) if len(ratio) > 0 and not np.all(np.isnan(ratio)) else 0

                peak_feature_list.append({
                    "session_number": session_id,
                    "peak_index": i,
                    "time": t[peak],
                    "peak_dfindex": peaks[i],
                    "ch4_reading": ch4[peak],
                    "co2_reading": co2[peak],
                    "prominence": pro[i],
                    "animal_number": session_data['Animal.Number'].iloc[0],
                    "device_number": session_data['Device_Number'].iloc[0],
                    "lactation_day": session_data['LactationDays'].iloc[0],
                    "session_length": session_length,
                    "peak_start_idx": start_idx[i],
                    "peak_end_idx": end_idx[i],
                    "start_end_diff": s_e_diff,
                    "right_left_ratio": right_left_ratio,
                    "left_right_ratio": left_right_ratio,
                    "peak_length": peak_length,
                    "peak_ratio": peak_ratio,
                    "peak_height_ratio": np.log1p(1/(peak_height * peak_length+ 1e-9)) if peak_height > 0 and peak_length > 0 else np.nan,
                    "peak_flat_std": peak_flat_std,
                    "peak_skewness": peak_skewness,
                    "peak_kurtosis": peak_kurtosis,
                    "ch4_co2_corr": corr,
                    "ch4_co2_ratio_mean": ratio_mean,
                    "ch4_co2_ratio_std": ratio_std
                })

    peak_features_df = pd.DataFrame(peak_feature_list)
    return peak_features_df


# --------------------------------------------------
# --------- Main Execution -----------
# --------------------------------------------------
def preprocess_data(org_df):

    # Load the data
    print("Loading data...")
    df=org_df

    # Data preprocessing
    print("Preprocessing data...")
    df['StartTime'] = pd.to_datetime(df['StartTime'], format='%d-%b-%Y %H:%M:%S', errors='coerce')
    df['EndTime'] = pd.to_datetime(df['EndTime'], format='%d-%b-%Y %H:%M:%S', errors='coerce')
    df['BoxTime'] = df['BoxTime'].str.replace(' sec', '', regex=False)
    df['BoxTime'] = pd.to_numeric(df['BoxTime'], errors='coerce').astype('Int64')
    df['CO2_Filter'] = df['CO2_Filter'] / 10000
    valid_sessions = df.groupby('session.ID').filter(lambda x: (x['CH4'] != 0).any())['session.ID'].unique()
    clean_df = df[df['session.ID'].isin(valid_sessions)]

    return clean_df

    # # Create peak features
    # print("Creating peak features...")
    # peak_features_df = create_peak_features(clean_df)

    # # Save to CSV
    # # Now df contains auc_t and accumulated auc for all sessions
    # from datetime import datetime
    # # today's date in yyyymmdd format
    # today_date = datetime.today().strftime("%Y%m%d")
    # # save CSV with date in filename
    # output_path = f"peak_features_{today_date}.csv"
    # peak_features_df.to_csv(output_path, index=False)

    # print(f"Peak features saved to: {output_path}")
    # print(f"Number of peaks detected: {len(peak_features_df)}")
    # print(f"Number of sessions processed: {df['session.ID'].nunique()}")