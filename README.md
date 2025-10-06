# Methane Emission Anomaly Detection Pipeline

An end-to-end pipeline for detecting anomalous methane emission peaks using unsupervised machine learning. This system processes raw sensor data, identifies emission peaks, filters anomalous measurements using Isolation Forest, and calculates AUC metrics for clean data.

## Pipeline workflow
Raw Sensor Data → Preprocessing → Peak Detection → Feature Extraction →
Anomaly Detection → AUC Calculation → Cleaned Data Output

## Directory Structure
iForest_Pipeline/
├── preprocess_function.py # Data preprocessing & feature extraction
├── iforest_model.py # Isolation Forest training & anomaly scoring
├── auc_calculation.py # AUC calculation for valid peaks
└── run_pipeline.py # Main pipeline execution script

### Feature Engineering Insights
- **Peak Boundary Features**: Heavily weighted by model to identify peaks with unreasonable start/end points
- **Inverse Ratios**: Use logarithmic inverses to make extreme values (very small peaks, very flat peaks) stand out
- **Temporal Context**: Features incorporate session-level context for normalized analysis

### Anomaly Threshold Selection
- **20% cutoff**: Based on empirical analysis of anomaly score distribution
- **Conservative filtering**: Retains 80% of peaks while removing most impactful noise
- **Domain-informed**: Balances data quality preservation with outlier removal
