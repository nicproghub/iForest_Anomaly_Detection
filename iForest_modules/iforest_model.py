import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import shap
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --------------------------------------------------
# --------- Visualization & Analysis ---------------
# --------------------------------------------------
def analyze_results(peak_features, X_scaled, available_features):
    """Analyze and visualize the results"""
    
    # 1. Distribution of anomaly scores
    plt.figure(figsize=(13, 10))
    
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(range(len(peak_features)), peak_features['anomaly_score'], 
                         c=peak_features['anomaly_score'], cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label='Anomaly Score')
    plt.xlabel('Session Index')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Scores by Session Index')
    plt.axhline(y=0, color='black', linestyle='--', label='Decision Boundary')
    
    # Histogram with score bins
    plt.subplot(2, 2, 2)
    score_bins = pd.cut(peak_features['anomaly_score'], bins=10)
    sns.histplot(data=peak_features, x='anomaly_score', hue=score_bins, 
                 multiple='stack', bins=30, legend=False)
    plt.title('Anomaly Score Distribution (Binned)')
    plt.axvline(x=0, color='black', linestyle='--')
    
        # Boxplot by score quartiles
    plt.subplot(2, 2, 3)
    peak_features['score_quartile'] = pd.qcut(peak_features['anomaly_score'], 
                                                q=4, labels=['Q1 (MostAnomalous)', 'Q2', 'Q3', 'Q4 (MostNormal)'])
    sns.boxplot(data=peak_features, x='score_quartile', y='anomaly_score')
    plt.title('Anomaly Score Distribution by Quartiles')
    plt.xticks(rotation=20)

        # Cumulative distribution
    plt.subplot(2, 2, 4)
    sorted_scores = np.sort(peak_features['anomaly_score'])
    plt.plot(sorted_scores, np.arange(len(sorted_scores)) / len(sorted_scores))
    plt.xlabel('Anomaly Score')
    plt.ylabel('Cumulative Proportion')
    plt.title('Cumulative Distribution of Anomaly Scores')
    plt.axvline(x=0, color='red', linestyle='--', label='Threshold (0)')
    plt.legend()

    plt.tight_layout(pad=3.0) 
    plt.show()
    
    # 2. PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=peak_features['anomaly_score'], 
                         cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label='Anomaly Score')
    plt.title('PCA Projection of peaks colored by Anomaly Score')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.show()
    
    # 3. Feature analysis for top anomalies
    top_anomalies = peak_features[peak_features['anomaly_label'] == -1].nsmallest(50, 'anomaly_score')
    bottom_anomalies = peak_features[peak_features['anomaly_label'] == 1].nlargest(50, 'anomaly_score')
    #print("\nTop 10 most anomalous peaks:")
    #print(top_anomalies[['session_number', 'peak_index','animal_number', 'device_number', 'anomaly_score']])
    
    return top_anomalies, bottom_anomalies

# --------------------------------------------------
# --------- Feature Importance Analysis ------------
# --------------------------------------------------
def analyze_feature_importance(model, peak_features, available_features, X_scaled):

   # Calculate how much each feature correlates with the final anomaly score
    feature_importances = []
    
    for feature in available_features:
        # For each feature, calculate its correlation with the anomaly score
        corr = np.corrcoef(peak_features[feature], peak_features['anomaly_score'])[0, 1]
        # Use absolute value since we care about strength of relationship, not direction
        feature_importances.append(abs(corr))

    importance_df = pd.DataFrame({
    'feature': available_features,
    'importance': feature_importances,
    'correlation_direction': [np.corrcoef(peak_features[feature], peak_features['anomaly_score'])[0, 1] 
                            for feature in available_features]
    }).sort_values('importance', ascending=False)
        
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Feature Importance (Mean Decrease in Accuracy)')
    plt.title('Features Importance for Anomaly Detection')
    plt.tight_layout()
    plt.show()
    
    return importance_df

# --------------------------------------------------
# --------- Main training function -----------------
# --------------------------------------------------

def train_isolation_forest(peak_features_df, features_to_use=None):

    # --- 2. Select features for model ---
    if features_to_use is None:
        features_to_use = [ 
            'start_end_diff','left_right_ratio', 'right_left_ratio',
            'peak_length', 'peak_ratio', 'peak_height_ratio', 
            'peak_flat_std', 'peak_skewness', 'ch4_co2_corr'
        ]

    X = peak_features_df[features_to_use].fillna(0)  # Fill NaNs if any

    # --- 3. Scale features  ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)  # keep original feature names

    # --- 4. Train Isolation Forest ---
    # contamination = expected proportion of outliers (can adjust based on domain knowledge)
    print(f"Training Isolation Forest with {len(features_to_use)} features...")
    # optimized parameters
    iso_forest = IsolationForest(
        n_estimators=150,        # Increased for stability
        max_samples=0.1,         # Better data utilization (1,381 samples/tree)
        contamination='auto',    # Let algorithm estimate
        max_features=0.8,        # Feature sub-sampling for diversity
        random_state=42,
        n_jobs=-1                # Parallel processing
    )

    iso_forest.fit(X_scaled)

    # ---- SHAP (for feature analysis)----
    explainer = shap.TreeExplainer(iso_forest)  # works for IsolationForest
    shap_values = explainer.shap_values(X_scaled)  # shape: (n_samples, n_features)
    # Convert to DataFrame
    shap_df = pd.DataFrame(shap_values, columns=X.columns)

    # --- 5. Predict anomalies ---
    # -1 = anomaly, 1 = normal
    peak_features_df['anomaly_label'] = iso_forest.predict(X_scaled)
    # actual continuous anomaly score
    peak_features_df['anomaly_score'] = iso_forest.decision_function(X_scaled)
    
    # --- 6. Inspect results ---
    # Peaks flagged as anomalies
    anomalous_peaks = peak_features_df[peak_features_df['anomaly_label'] == -1]

    print(f"Detected {len(anomalous_peaks)} anomalous peaks out of {len(peak_features_df)} total peaks")


    # --- 7. Visualize and analyze results ---
    # plot anomaly scores
    top_anomalies, bottom_anomalies = analyze_results(peak_features_df, X_scaled, features_to_use)    
    # Analyze feature importance
    importance_df  = analyze_feature_importance(iso_forest, peak_features_df, features_to_use, X_scaled)
    # Add to original dataframe (optional)
    peak_features_df = pd.concat([peak_features_df.reset_index(drop=True), shap_df], axis=1)
    # Add to original dataframe (optional)
    # peak_features_df = pd.concat([peak_features_df.reset_index(drop=True), shap_df], axis=1)
    # peak_features_df.to_csv("peak_features_with_shap_values.csv", index=False)

    # Print top 3 important features
    print(f"Top 3 most important features: {importance_df['feature'][:3].tolist()}")

    return peak_features_df,importance_df


def get_anomalous_peaks(peak_features_df, anomaly_threshold=0.2):
    """Get top anomalous peaks based on threshold"""
    # Sort by anomaly_score (ascending - most negative first)
    all_peaks_sorted = peak_features_df.sort_values('anomaly_score', ascending=True).reset_index(drop=True)

    # Calculate number of peaks to select
    top_count = int(len(all_peaks_sorted) * anomaly_threshold)
    
    # Select top anomalous peaks
    top_peaks = all_peaks_sorted.head(top_count)[['session_number', 'peak_index', 'anomaly_score']]
    top_peaks_set = set(zip(top_peaks['session_number'], top_peaks['peak_index']))
    
    return top_peaks, top_peaks_set