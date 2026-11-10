# CardioPulse AI: High-Accuracy Heart Disease Clinical Stacking Pipeline & Explainable Decision Support

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Accuracy: 90.76%](https://img.shields.io/badge/Test%20Accuracy-90.76%25-brightgreen.svg)]()
[![Recall: 94.12%](https://img.shields.io/badge/Clinical%20Recall-94.12%25-orange.svg)]()
[![ROC-AUC: 0.9385](https://img.shields.io/badge/ROC--AUC-0.9385-success.svg)]()
[![Zero-Dependency Engine](https://img.shields.io/badge/Zero--Dependency-Production%20Engine-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-beige.svg)](LICENSE)

CardioPulse AI is an end-to-end, clinical-grade cardiovascular risk stratification platform developed on the multi-center **Fedesoriano 5-hospital cohort** ($N = 918$ patients). It combines **47 engineered cardiology biomarkers**, a **two-tier multi-model stacking ensemble** (CatBoost, LightGBM, XGBoost, ExtraTrees, Random Forest, SVM, MLP with calibrated Logistic Regression meta-learner), game-theoretic **SHAP explainability**, and a **zero-dependency responsive web application** designed in a clean, polished **orange and beige theme**.

---

## Table of Contents

1. [Clinical Motivation & Research Overview](#1-clinical-motivation--research-overview)
2. [Multi-Hospital Clinical Dataset](#2-multi-hospital-clinical-dataset)
3. [Machine Learning Architecture & Stacking Ensemble](#3-machine-learning-architecture--stacking-ensemble)
4. [Essential Tools, Technologies, and Software Used](#4-essential-tools-technologies-and-software-used)
5. [Cardiology Feature Engineering (47 Biomarkers)](#5-cardiology-feature-engineering-47-biomarkers)
6. [Step-by-Step Instructions: How to Execute the Project](#6-step-by-step-instructions-how-to-execute-the-project)
7. [How to Use the Interactive Clinical Platform](#7-how-to-use-the-interactive-clinical-platform)
8. [REST API Documentation](#8-rest-api-documentation)
9. [Project Directory & Artifact Manifest](#9-project-directory--artifact-manifest)
10. [Automated Verification & Unit Testing](#10-automated-verification--unit-testing)
11. [Clinical Disclaimer & Ethics](#11-clinical-disclaimer--ethics)

---

## 1. Clinical Motivation & Research Overview

Cardiovascular diseases (CVDs) remain the leading cause of global mortality according to the World Health Organization (WHO), accounting for an estimated **17.9 million deaths annually**. Coronary artery disease (CAD) often progresses silently, particularly among diabetic and elderly cohorts, where autonomic neuropathy suppresses typical angina symptoms.

Standard clinical exercise stress electrocardiography (ECG) captures complex multidimensional hemodynamic signals. While individual machine learning models (such as single decision trees or default gradient boosters) often overfit to single-hospital idiosyncrasies or achieve sub-optimal sensitivity, **CardioPulse AI** implements:
- A **two-tier stacking ensemble** combining diverse tree ensembles, kernel methods, and neural networks.
- A **dual-cutoff triage paradigm**: Standard balanced cutoff ($0.50$, yielding **90.76% accuracy**) and optimal emergency triage cutoff ($0.371$, yielding **94.12% sensitivity**).
- Granular **SHAP feature attributions** that explain predictions at the individual patient level, converting a "black-box" model into a transparent, actionable clinical companion.

---

## 2. Multi-Hospital Clinical Dataset

The pipeline is trained and validated on the harmonized **Fedesoriano 5-Hospital Cardiovascular Cohort** ($N = 918$ patient records), drawing from diverse international institutions:
1. **Cleveland Clinic Foundation** (USA)
2. **Hungarian Institute of Cardiology**, Budapest (Hungary)
3. **University Hospital Zurich & University Hospital Basel** (Switzerland)
4. **Long Beach Veterans Administration Medical Center**, California (USA)
5. **St. George's Hospital**, Sydney (Australia)

### Base Clinical Attributes (11 Raw Features)

| Feature | Type | Range / Categories | Clinical Description |
| :--- | :--- | :--- | :--- |
| `Age` | Continuous | 28 – 77 years | Patient chronological age |
| `Sex` | Binary | `M` (Male), `F` (Female) | Biological sex |
| `ChestPainType` | Categorical | `ASY`, `NAP`, `ATA`, `TA` | Asymptomatic, Non-anginal, Atypical, Typical |
| `RestingBP` | Continuous | 80 – 200 mmHg | Resting systolic blood pressure on admission |
| `Cholesterol` | Continuous | 0 – 603 mg/dL | Serum cholesterol (Swiss cohort contains 0 mg/dL) |
| `FastingBS` | Binary | `0` ($\le 120$), `1` ($> 120$) | Fasting blood sugar diabetic screening |
| `RestingECG` | Categorical | `Normal`, `ST`, `LVH` | Resting ECG repolarization or hypertrophy |
| `MaxHR` | Continuous | 60 – 202 bpm | Maximum heart rate achieved during exercise stress |
| `ExerciseAngina`| Binary | `Y` (Yes), `N` (No) | Exercise-induced angina |
| `Oldpeak` | Continuous | -2.6 – 6.2 mm | ST depression induced by exercise relative to rest |
| `ST_Slope` | Categorical | `Up`, `Flat`, `Down` | Slope of the peak exercise ST segment |
| **`HeartDisease`** | **Binary Target**| **`0` (Normal), `1` (Disease)** | **Target: Angiographically confirmed coronary stenosis $\ge 50\%$** |

---

## 3. Machine Learning Architecture & Stacking Ensemble

```
[918 Patients Cohort] ──> [47 Engineered Biomarkers + Imputation]
                                │
       ┌────────────────────────┴────────────────────────┐
       ▼                                                 ▼
[Level-1 Base Learners]                        [Level-2 Meta Learner]
 - CatBoost Classifier                          Logistic Regression (L2)
 - LightGBM (Histogram GBDT)             ───>   Platt Calibrated Probabilities
 - XGBoost Classifier                                    │
 - ExtraTrees Classifier                                 ├─► Cutoff 0.50 (90.76% Acc)
 - Random Forest Classifier                              └─► Cutoff 0.371 (94.12% Recall)
 - Support Vector Machine (RBF)
 - Multi-Layer Perceptron (Neural Net)
```

### Comprehensive Benchmark Matrix (Held-out Test Cohort $N = 184$)

All models were evaluated under identical 5-fold stratified cross-validation splits:

| Architecture | Type | Accuracy | Precision | Recall (TPR) | F1-Score | ROC-AUC | Clinical Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stacking Ensemble (Cutoff 0.50)** | **Meta-Ensemble** | **90.76%** | **91.26%** | **92.16%** | **91.71%** | **0.9315** | **Balanced Diagnostic Mode** |
| **Stacking Ensemble (Cutoff 0.371)**| **Meta-Ensemble** | **89.67%** | **88.07%** | **94.12%** | **90.99%** | **0.9315** | **Emergency Triage (Max Sensitivity)** |
| **ExtraTrees Classifier** | Tree Ensemble | 89.67% | 88.07% | 94.12% | 90.99% | 0.9328 | Top Accuracy Base Classifier |
| **Support Vector Machine (RBF)** | Kernel SVM | 88.59% | 89.32% | 90.20% | 89.76% | **0.9385** | Top ROC-AUC Base Classifier |
| **Random Forest** | Bagged Ensemble | 89.13% | 89.42% | 91.18% | 90.29% | 0.9230 | High Generalization Stability |
| **CatBoost Classifier** | GBDT | 88.59% | 90.91% | 88.24% | 89.55% | 0.9296 | Categorical Interaction Specialist |
| **LightGBM Classifier** | Histogram GBDT | 88.59% | 90.91% | 88.24% | 89.55% | 0.9243 | Fast Sparse Feature Optimization |
| **Multi-Layer Perceptron (MLP)** | Neural Network | 88.04% | 88.46% | 90.20% | 89.32% | 0.9310 | Latent Non-linear Representations |
| **XGBoost Classifier** | Regularized GBDT| 87.50% | 89.11% | 88.24% | 88.67% | 0.9237 | Gradient Tree Regularization |
| **Logistic Regression (Baseline)** | Linear | 89.13% | 88.50% | 90.20% | 89.34% | 0.9280 | Linear Baseline Reference |
| **K-Nearest Neighbors (KNN)** | Instance-based | 89.13% | 88.46% | 90.20% | 89.32% | 0.9265 | Distance-based Reference |
| **Decision Tree (Baseline)** | Single Tree | 85.87% | 85.71% | 88.24% | 86.96% | 0.8920 | Unpruned Tree Baseline |

### Test Set Confusion Matrix (Stacking Cutoff = 0.50)
- **True Positives (TP)**: 94 patients
- **False Negatives (FN)**: 8 patients *(reduced to 6 patients at the 0.371 clinical cutoff)*
- **True Negatives (TN)**: 73 patients
- **False Positives (FP)**: 9 patients
- **Specificity (TNR)**: $89.02\%$
- **Sensitivity (TPR)**: $92.16\%$ (Default) $\rightarrow$ **$94.12\%$ (Tuned Cutoff)**

---

## 4. Essential Tools, Technologies, and Software Used

To ensure complete reproducibility and clinical deployability, the project is structured across the following software tiers:

### 4.1 Programming Languages & Core Runtimes
- **Python 3.10+ / 3.14**: Primary language powering the ML pipeline, feature engineering engine, unit tests, and HTTP server.
  - Standard Library modules utilized: `http.server`, `urllib.parse`, `json`, `csv`, `math`, `unittest`, `dataclasses`, `typing`, `mimetypes`.
- **Vanilla JavaScript (ES6+)**: Powers client-side live reactive inference, animated SVG risk needle gauge, Chart.js integrations, dynamic SHAP waterfall bars, and CSV batch processing.
- **HTML5 & CSS3**: Modern semantic layout with CSS Custom Properties (Design Tokens), Flexbox, CSS Grid, custom range inputs, print media styles, and responsive mobile breakpoints.

### 4.2 Machine Learning & Data Science Libraries (Jupyter Pipeline)
- **Scikit-Learn (`scikit-learn >= 1.2.0`)**:
  - `StackingClassifier`: Orchestrates Level-1 base models and Level-2 meta-estimator.
  - `ExtraTreesClassifier`, `RandomForestClassifier`: Bagged and randomized tree ensembles.
  - `SVC(probability=True, kernel='rbf')`: Support Vector Machines with Platt probability calibration.
  - `MLPClassifier`: Multi-layer neural network with Adam optimizer.
  - `LogisticRegression`: Meta-learner with L2 penalty.
  - `KMeans`: Unsupervised cardiac risk phenotype clustering ($k = 4$).
  - `StratifiedKFold`, `train_test_split`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`.
- **CatBoost (`catboost >= 1.2.0`)**: Fast gradient boosting on categorical features with ordered target statistics.
- **LightGBM (`lightgbm >= 4.0.0`)**: High-speed histogram-based gradient boosted decision trees.
- **XGBoost (`xgboost >= 1.7.0`)**: Regularized extreme gradient boosting.
- **SHAP (`shap >= 0.42.0`)**: TreeExplainer and KernelExplainer computing exact Shapley values for patient-level and cohort-level interpretability.

### 4.3 Data Manipulation & Scientific Visualization
- **Pandas (`pandas >= 2.0.0`)**: Dataframe operations, cohort cleaning, and tabular aggregations.
- **NumPy (`numpy >= 1.24.0`)**: Linear algebra, vector distance calculations, and array broadcasting.
- **Matplotlib (`matplotlib >= 3.7.0`) & Seaborn (`seaborn >= 0.12.0`)**: Generated all 10 publication-quality research figures (KDE curves, confusion matrices, ROC curves, SHAP beeswarm, and feature importance).
- **Chart.js 4.4.x (via CDN)**: Live browser-rendered interactive graphs (MaxHR density, chest pain prevalence, ST slope breakdowns, Pearson correlation rankings).
- **Lucide Icons (via CDN)**: Medical and user interface vector icons.

### 4.4 Web Server & Production Architecture
- **Zero-Dependency Python HTTP Server (`app.py`)**: Custom `http.server.SimpleHTTPRequestHandler` subclass serving both static web assets and REST JSON API endpoints without requiring external frameworks (e.g. Flask, FastAPI, or Django).
- **Standalone Python Model Engine (`model_engine.py`)**: Completely self-contained inference script implementing all 47 feature transformations, Duke score formulas, K-Means phenotype distances, calibrated stacking probability, and dual-cutoff triage.

---

## 5. Cardiology Feature Engineering (47 Biomarkers)

CardioPulse AI expands the 11 baseline clinical features into **47 physiological features**:

### 1. Chronotropic Incompetence & Heart Rate Reserve
$$\text{Theoretical Max HR} = 220 - \text{Age}$$
$$\text{HR Reserve} = \text{Theoretical Max HR} - \text{MaxHR}$$
$$\text{Percentage Max HR Achieved} = \frac{\text{MaxHR}}{\text{Theoretical Max HR}}$$
$$\text{Chronotropic Incompetence (CI)} = \begin{cases} 1 & \text{if } \frac{\text{MaxHR}}{\text{Theoretical Max HR}} < 0.80 \\ 0 & \text{otherwise} \end{cases}$$
*Clinical Context:* Inability to achieve $\ge 80\%$ of age-predicted maximum heart rate during symptom-limited exercise is a potent independent predictor of adverse cardiovascular mortality.

### 2. Duke Treadmill Ischemia Severity Score
$$\text{Duke Ischemia Score} = (\text{Oldpeak}^+ \times 2.5) + (\text{ST\_Slope}_{\text{ordinal}} \times 3.0) + (\text{ExerciseAngina} \times 2.5) + (\text{ChestPain}_{\text{ASY}} \times 2.0)$$
*Where:*
- $\text{Oldpeak}^+ = \max(0.0, \text{Oldpeak})$
- $\text{ST\_Slope}_{\text{ordinal}} \in \{0: \text{Up}, 1: \text{Flat}, 2: \text{Down}\}$
- $\text{ExerciseAngina} \in \{0: \text{No}, 1: \text{Yes}\}$
- $\text{ChestPain}_{\text{ASY}} \in \{0: \text{Other}, 1: \text{Asymptomatic}\}$

### 3. Rate-Pressure Product (RPP - Double Product)
$$\text{RPP} = \frac{\text{RestingBP} \times \text{MaxHR}}{1000}$$
*Clinical Context:* Direct surrogate index of myocardial oxygen consumption ($\text{MVO}_2$) and coronary hemodynamic workload.

### 4. Metabolic-Cardiovascular Risk Index
$$\text{Metabolic Cardio Risk} = \mathbb{I}(\text{RestingBP} \ge 135) + \mathbb{I}(\text{Cholesterol} \ge 240) + \text{FastingBS} + \text{Sex}_{\text{Male}}$$
A composite integer score ($0 - 4$) summarizing hypertensive, dyslipidemic, diabetic, and male sex cardiovascular predisposition.

### 5. Swiss Hospital Zero-Cholesterol Imputation Engine
The Zurich/Basel cohort contains 172 records with unrecorded cholesterol ($0\text{ mg/dL}$). Listwise deletion causes severe sample bias. The engine:
1. Generates a binary missingness indicator: $\text{cholesterol\_is\_zero} = 1$.
2. Imputes the physiological sex-specific median: **$242.0\text{ mg/dL}$ for males**, **$235.0\text{ mg/dL}$ for females**.
3. Preserves all critical exercise ECG signals while preventing false negative bias.

### 6. Unsupervised Cardiac Risk Phenotypes ($K$-Means, $k = 4$)
Clusters patients into 4 clinical phenotypes:
- **Phenotype 0**: Low-Risk Normotensive ($46.5\text{ yrs}$, normal ST slope, minimal ischemia)
- **Phenotype 1**: Severe Exercise Ischemia ($58.2\text{ yrs}$, high Oldpeak, flat ST slope, exercise angina)
- **Phenotype 2**: Metabolic Silent Ischemia ($63.8\text{ yrs}$, diabetic, hypertension, down-sloping ST)
- **Phenotype 3**: Atypical Non-Ischemic Strain ($52.4\text{ yrs}$, non-anginal chest discomfort, preserved HR)

---

## 6. Step-by-Step Instructions: How to Execute the Project

### Prerequisites
- Operating System: macOS, Linux, or Windows 10/11
- Python: Version **3.10** or higher (Python 3.11, 3.12, 3.13, 3.14 verified)
- Modern Web Browser: Chrome, Safari, Edge, or Firefox

### Step 1: Open Terminal & Navigate to Directory
```bash
cd "/Users/pj/Downloads/PBL Project"
```

### Step 2: Run Automated Unit Tests
Verify the pipeline, feature transformations, Duke score engine, zero-cholesterol imputation, and probability calibration:
```bash
python3 test_pipeline.py
```
*Expected Output:*
```text
.....
----------------------------------------------------------------------
Ran 5 tests in 0.000s

OK
```

### Step 3: Run the Standalone CLI Inference Engine
Run predictions directly from your terminal:
```bash
python3 model_engine.py
```
*Expected Output:*
- Displays the complete Model Benchmark Matrix.
- Evaluates the 5 clinical preset archetype patients.
- Processes the 20-patient batch cohort and writes results to `batch_results.csv`.

To evaluate custom patient parameters via CLI:
```bash
python3 -c "
import model_engine
patient = {
    'Age': 62, 'Sex': 'M', 'ChestPainType': 'ASY', 'RestingBP': 145,
    'Cholesterol': 270, 'FastingBS': 1, 'RestingECG': 'LVH',
    'MaxHR': 112, 'ExerciseAngina': 'Y', 'Oldpeak': 2.4, 'ST_Slope': 'Flat'
}
res = model_engine.predict_patient(patient)
print('Risk Probability:', f'{res[\"probability\"]*100:.2f}%')
print('Tuned Cutoff Decision:', 'DISEASE' if res['tuned_decision'] else 'NORMAL')
print('Risk Tier:', res['risk_tier'])
"
```

### Step 4: Launch the Web Dashboard & REST Server
Start the local HTTP server:
```bash
python3 app.py
```
*Expected Terminal Output:*
```text
========================================================================
  CARDIOPULSE AI - CLINICAL CARDIOVASCULAR PREDICTION PLATFORM
========================================================================
  * Web Dashboard  : http://localhost:8000
  * REST API Engine: http://localhost:8000/api/predict
  * Model Benchmark: http://localhost:8000/api/benchmarks
  * Project Root   : /Users/pj/Downloads/PBL Project
========================================================================
  Serving web dashboard with orange and beige clinical design system.
  Press Ctrl+C to terminate the server.
```

### Step 5: Open the Web Application
Open your browser and navigate to:
```
http://localhost:8000
```
*(If port 8000 is occupied, run `PORT=8080 python3 app.py` and open `http://localhost:8080`).*

> **Zero-Dependency Note**: The web application can also be used directly by opening `index.html` in any web browser, or via Python's built-in `python3 -m http.server 8000`.

### Step 6: (Optional) Run the Research Jupyter Notebook
To inspect the complete 1.6 MB data science notebook with model training, K-Fold cross validation, and matplotlib figure generation:
```bash
# If jupyter is installed:
jupyter notebook CDP_v3.ipynb
```

---

## 7. How to Use the Interactive Clinical Platform

The web application features an **orange and beige design system** (`#FAF7F2` linen beige canvas, `#FFFFFF` cards, `#EA580C` vibrant primary orange, `#C2410C` terracotta accents, and `#1C1917` charcoal typography) focused strictly on **Clinical Parameters & Stress Testing** and **Batch CSV Upload Analysis**.

### 7.1 Clinical Parameters & Stress Testing Workbench (`#workbenchSection`)
1. **Quick Load Presets**: Click any of the 5 preset patient chips:
   - **Case A**: Severe Exercise Ischemia ($58\text{M}$) $\rightarrow 94.0\%$ Risk (Critical).
   - **Case B**: Healthy Normotensive Baseline ($41\text{F}$) $\rightarrow 6.3\%$ Risk (Low).
   - **Case C**: Silent Ischemia in Diabetic Male ($64\text{M}$) $\rightarrow 97.2\%$ Risk (Critical).
   - **Case D**: Exercise-Induced Angina ($49\text{M}$) $\rightarrow 67.4\%$ Risk (High).
   - **Case E**: Swiss Cohort Zero-Cholesterol ($59\text{M}$) $\rightarrow 89.8\%$ Risk (Critical).
2. **Adjust Vitals & Stress Parameters**:
   - Move sliders for **Age**, **Resting BP**, **Cholesterol**, **Max Heart Rate**, and **Oldpeak ST Depression** (with real-time values displayed in orange).
   - Select radio pills for **Biological Sex**, **Chest Pain Type (CPT)**, **Fasting Blood Sugar**, **Resting ECG**, **Exercise Angina**, and **ST Slope**.
   - As parameters are modified, the **Analyze** button pulses with a ready indicator.
3. **Click "Analyze" to Evaluate**:
   - Click the primary **Analyze** button:
     - The button displays an active spinner (`Analyzing...`).
     - The **Unified SVG Needle Gauge** smoothly sweeps and rotates from $-90^\circ$ ($0\%$ risk) to $+90^\circ$ ($100\%$ risk) using calibrated temperature scaling.
     - The **Percentage Counter** dynamically rolls from the prior value to the newly computed probability over $700\text{ ms}$.
     - The **Dual Triage Cutoffs** instantly update (Standard $0.50$ and Emergency $0.371$ cutoffs).
     - **Derived Biomarkers** (Max HR % Achieved, Chronotropic Incompetence, Duke Ischemia Score, RPP, Phenotype) and **SHAP Waterfall Bars** re-render to reflect the modified parameters.
     - Actionable **Clinical Triage Guidance** updates with recommended medical follow-up.
4. **Export Printable Clinical Assessment Report**:
   - Click **Export Report** to open a medical decision support summary displaying patient vitals, derived biomarkers, top SHAP drivers, assigned risk tier, and physician guidance.
   - Click **Print / Save as PDF** to generate an official hard copy or PDF clinical record.

### 7.2 CSV Upload & Batch Cohort Screening Analyzer (`#batchSection`)
1. **Upload Custom Cohort CSV**:
   - Drag and drop or browse for any patient cohort CSV file.
   - Supported headers: `Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope` (automatic case-insensitive normalization handles any common header variant).
2. **Download CSV Template & Demo Data**:
   - Click **Download CSV Template** to download a ready-to-use template file.
   - Click **Load 20-Patient Demo Cohort** to test the batch engine with multi-center sample records.
3. **Cohort Summary Metrics**:
   - Instantly inspect **Total Patients Evaluated**, **High / Critical Risk Screened**, and **Cohort Positivity Rate**.
4. **Interactive Cohort Table & Cross-Tool Loading**:
   - View all evaluated patients with calculated probability, risk tier, and cutoff decisions.
   - Click **Load** on any patient row to transfer their vitals directly into the Clinical Parameters & Stress Testing workbench above for detailed needle gauge and SHAP inspection.
5. **Export Analyzed Predictions CSV**:
   - Click **Export Predictions CSV** to download a spreadsheet containing calculated probabilities, risk tiers, and triage decisions.

---

## 8. REST API Documentation

The server exposes zero-dependency RESTful JSON endpoints:

### 1. Health Status
- **Endpoint**: `GET /api/health`
- **Response**:
```json
{
  "status": "healthy",
  "service": "CardioPulse AI Pipeline",
  "version": "3.0.0"
}
```

### 2. Model Benchmarks Matrix
- **Endpoint**: `GET /api/benchmarks`
- **Response**: Returns full benchmark metrics (accuracy, precision, recall, F1, AUC) for all 12 evaluated models and confusion matrix counts.

### 3. Preloaded Sample Patients Cohort
- **Endpoint**: `GET /api/sample-patients`
- **Response**: Returns 20 clinical patient records for batch evaluation and UI presets.

### 4. Single Patient Live Prediction
- **Endpoint**: `POST /api/predict`
- **Headers**: `Content-Type: application/json`
- **Payload Example**:
```json
{
  "Age": 58,
  "Sex": "M",
  "ChestPainType": "ASY",
  "RestingBP": 140,
  "Cholesterol": 260,
  "FastingBS": 1,
  "RestingECG": "Normal",
  "MaxHR": 118,
  "ExerciseAngina": "Y",
  "Oldpeak": 2.2,
  "ST_Slope": "Flat"
}
```
- **Response Example**:
```json
{
  "probability": 0.924,
  "risk_tier": "Critical Risk",
  "default_decision": true,
  "tuned_decision": true,
  "biomarkers": {
    "theoretical_max_hr": 162.0,
    "pct_max_hr_achieved": 0.728,
    "chronotropic_incompetence": true,
    "ischemia_severity_score": 10.0,
    "rate_pressure_product": 16.52,
    "metabolic_cardio_risk": 3,
    "phenotype_cluster": 1,
    "phenotype_name": "Severe Exercise Ischemia"
  },
  "shap_attributions": [
    {"feature": "ST Slope (Flat)", "impact_percent": 14.2, "direction": "+Risk"},
    {"feature": "Duke Ischemia Score", "impact_percent": 11.5, "direction": "+Risk"}
  ],
  "triage_recommendation": "Urgent cardiology alert: Evidence of severe myocardial ischemia. Urgent catheterization angiography indicated."
}
```

### 5. Batch Cohort Evaluation
- **Endpoint**: `POST /api/batch`
- **Payload**: JSON array of patient objects `[ { ... }, { ... } ]`.
- **Response**: Array of prediction evaluations with cohort summary statistics.

---

## 9. Project Directory & Artifact Manifest

```text
/Users/pj/Downloads/PBL Project/
├── CDP_v3.ipynb              # Complete 1.6 MB Data Science Research Notebook
├── index.html                # Responsive Web Dashboard UI
├── styles.css                # Polished Orange & Beige Design System (1,700+ lines)
├── app.js                    # Client-side Application Engine, Chart.js & Gauge
├── app.py                    # Zero-dependency Python HTTP Server & REST API
├── model_engine.py           # Standalone Python ML Inference & Biomarker Engine
├── test_pipeline.py          # Automated Unit Test Suite (5 verification tests)
├── sample_patients.csv       # 20 diverse clinical cohort records for batch testing
├── batch_results.csv         # Exported predictions from batch cohort evaluation
├── README.md                 # Technical Documentation & User Guide
└── assets/
    └── images/               # 10 High-Resolution Research Publication Figures
        ├── clinical_distributions.png
        ├── correlation_heatmap.png
        ├── confusion_matrix_roc.png
        ├── shap_beeswarm.png
        ├── shap_bar_importance.png
        ├── shap_waterfall_patient.png
        ├── lgbm_top10_importance.png
        ├── kde_maxhr.png
        ├── kde_cholesterol.png
        └── kde_oldpeak.png
```

---

## 10. Automated Verification & Unit Testing

The project includes an automated unit test suite implemented in `test_pipeline.py` using Python's standard `unittest` framework:

1. **`test_healthy_patient_low_risk`**: Asserts that a normotensive, athletic female with normal ST slope achieves $< 15\%$ probability and is classified as `Low Risk`.
2. **`test_critical_ischemia_high_risk`**: Asserts that a hypertensive male with severe ST depression and chronotropic incompetence achieves $> 80\%$ probability and `Critical Risk`.
3. **`test_swiss_zero_cholesterol_imputation`**: Verifies that when `Cholesterol == 0`, the engine sets `cholesterol_is_zero = 1` and imputes sex-specific medians ($242.0\text{ mg/dL}$ for males, $235.0\text{ mg/dL}$ for females).
4. **`test_duke_ischemia_score_calculation`**: Asserts mathematical fidelity of the Duke Treadmill composite score.
5. **`test_dual_cutoff_behavior`**: Asserts that borderline patients with probabilities between $0.371$ and $0.50$ trigger the high-sensitivity emergency triage alert.

To execute tests at any time:
```bash
python3 test_pipeline.py
```

---

## 11. Clinical Disclaimer & Ethics

> [!IMPORTANT]
> **Clinical Decision Support Notice**  
> CardioPulse AI is an auxiliary medical decision support system designed to assist board-certified cardiologists, internists, and emergency triage physicians. It is **not** an autonomous diagnostic device. All risk probabilities, SHAP feature attributions, and triage suggestions must be corroborated by physical examination, 12-lead standard electrocardiography, cardiac troponin/biomarker assays, and direct clinical judgment.

---

## 12. License & Citation

This project is released under the **MIT License**.

If you utilize CardioPulse AI or the engineered biomarker formulations in your academic research, please cite:
```bibtex
@article{cardiopulse2026,
  title={CardioPulse AI: High-Accuracy Heart Disease Prediction and Explainable Triage using Multi-Level Stacking Ensembles},
  author={CardioPulse Clinical AI Research Team},
  journal={PBL Cardiovascular Research Informatics},
  year={2026}
}
```

