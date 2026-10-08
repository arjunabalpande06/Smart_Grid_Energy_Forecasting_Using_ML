# Electricity Demand & Load Forecasting (PBL Machine Learning Project)

A portable, cross-platform machine learning and deep learning project for forecasting electrical energy demand across household and national grid scales.

> **Note:** **This project runs completely offline and locally.** All datasets, pre-trained model artifacts, training workflows, and Gradio applications run on your local machine using standard Python and VS Code — no Google Colab, Google Drive, or cloud subscriptions required.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Folder Structure](#-folder-structure)
- [Dataset & Domain Comparison](#-dataset--domain-comparison)
- [Folder 1: FRANCE_DATASET (Household Power Consumption)](#-folder-1-france_dataset-household-power-consumption)
  - [Overview & Raw Data](#france-overview--raw-data)
  - [Notebook Architecture](#france-notebook-architecture)
  - [Feature Engineering Strategies](#france-feature-engineering-strategies)
  - [Model Performance Benchmark](#france-model-performance-benchmark)
  - [Gradio Interactive Interfaces](#france-gradio-interactive-interfaces)
  - [Saved Artifacts & Outputs](#france-saved-artifacts--outputs)
- [Folder 2: INDIAN_DATASET (National Electricity Demand)](#-folder-2-indian_dataset-national-electricity-demand)
  - [Overview & Raw Data](#india-overview--raw-data)
  - [Notebook Architecture](#india-notebook-architecture)
  - [Feature Engineering Strategies](#india-feature-engineering-strategies)
  - [Model Performance Benchmark](#india-model-performance-benchmark)
  - [Gradio Interactive Interfaces](#india-gradio-interactive-interfaces)
  - [Saved Artifacts & Outputs](#india-saved-artifacts--outputs)
- [Machine Learning & Deep Learning Architectures](#-machine-learning--deep-learning-architectures)
- [Environment Setup & Installation](#-environment-setup--installation)
  - [Python Version](#python-version)
  - [Virtual Environment Setup](#virtual-environment-setup)
  - [Installing Requirements](#installing-requirements)
- [VS Code & Jupyter Setup](#-vs-code--jupyter-setup)
- [Execution (CPU & GPU)](#-execution-cpu--gpu)
- [Running Gradio Web Applications](#-running-gradio-web-applications)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🔍 Project Overview

Accurate electrical load forecasting is critical for grid stability, peak-load management, energy trading, and household energy optimization. This project provides a comparative machine learning and deep learning study across two distinct temporal and operational domains:

1. **Micro-Scale Household Level (`FRANCE_DATASET`)**:
   - Predicts 1-minute global active power consumption (kW) for an individual household in Sceaux, France.
   - Leverages high-frequency lag features, temporal cycles, and rolling sequence windows.
2. **Macro-Scale National Grid Level (`INDIAN_DATASET`)**:
   - Forecasts national hourly electricity demand (MW) across India.
   - Evaluates calendar temporal encoding against deep sequential recurrent modeling (LSTM).

Both domains feature interactive web applications built with **Gradio**, enabling real-time inference directly in your browser.

---

## 📂 Folder Structure

The repository is modularly organized into two self-contained dataset directories alongside global environment configurations:

```
PBL_Model/
│
├── FRANCE_DATASET/
│   ├── household_power_consumption.csv          # Raw dataset (~2.075M rows, semicolon-delimited)
│   │
│   ├── Linear Regression.ipynb                  # Linear Regression training & evaluation (physical/temporal features)
│   ├── LR_Gradio.ipynb                          # Linear Regression training & Gradio UI (lag features)
│   ├── LSTM.ipynb                               # LSTM sequential training & evaluation (30-step window)
│   ├── LSTM_Gradio.ipynb                        # LSTM training & Gradio UI (60-step test index inference)
│   ├── RFR.ipynb                                # Random Forest Regressor training & evaluation (30-step window)
│   ├── RFR_Gradio.ipynb                         # Random Forest Regressor training & Gradio UI (lag features)
│   ├── SVR.ipynb                                # Support Vector Regressor training & evaluation (30-step window)
│   ├── SVR_Gradio.ipynb                         # Support Vector Regressor training & Gradio UI (lag features)
│   ├── XGBoost.ipynb                            # XGBoost Regressor training & evaluation (30-step window)
│   ├── XGBoost_Gradio.ipynb                     # XGBoost Regressor training & Gradio UI (lag features)
│   │
│   ├── Linear_Regression_Output/
│   │   └── linear_regression_energy_model.pkl   # Serialized Linear Regression model
│   ├── LSTM_Output/
│   │   ├── linear_regression_energy_model.pkl   # Serialized Linear Regression model
│   │   ├── smart_grid_lstm_model.keras          # Trained Keras LSTM neural network
│   │   └── lstm_predictions.csv                 # Test set actual vs. predicted values
│   ├── RFR_Outputs/
│   │   ├── random_forest_model.pkl              # Serialized Random Forest Regressor (~18 MB)
│   │   └── random_forest_predictions.csv        # Test set predictions
│   ├── SVR_Output/
│   │   ├── smart_grid_svr_model.pkl             # Serialized Support Vector Regressor (~1.8 MB)
│   │   └── svr_predictions.csv                  # Test set predictions
│   └── XGBoost_Output/
│       ├── xgboost_model.pkl                    # Serialized XGBoost model (~282 KB)
│       └── xgboost_predictions.csv              # Test set predictions
│
├── INDIAN_DATASET/
│   ├── dataset/
│   │   ├── hourlyLoadDataIndia.xlsx             # Primary hourly national load dataset (Excel, ~3.3 MB)
│   │   └── monthly_temp.xlsx                    # Auxiliary monthly temperature dataset
│   │
│   ├── Linear Regression Model Data/
│   │   └── linear_regression_model.pkl          # Serialized Linear Regression model
│   ├── RFR Model Data/
│   │   └── random_forest_model.pkl              # Serialized Random Forest model (~302 MB)
│   ├── SVR Model Data/
│   │   ├── svr_model.pkl                        # Serialized Support Vector Regressor (~2.5 MB)
│   │   └── svr_scaler.pkl                       # Serialized StandardScaler for SVR feature scaling
│   │
│   ├── LR_Gradio.ipynb                          # Linear Regression pipeline & Gradio UI (all-in-one)
│   ├── LSTM_Gradio.ipynb                        # 2-Layer LSTM deep neural network & Gradio UI (all-in-one)
│   ├── RFR_Gradio.ipynb                         # Random Forest pipeline & Gradio UI (all-in-one)
│   ├── SVR_Gradio.ipynb                         # SVR pipeline (with StandardScaler) & Gradio UI (all-in-one)
│   └── XGBoost_Gradio.ipynb                     # XGBoost pipeline & Gradio UI (all-in-one)
│
├── requirements.txt                             # Unified Python package dependencies
├── README.md                                    # Comprehensive project documentation
└── .gitignore                                   # Standard Python / Jupyter / model artifact ignore rules
```

---

## ⚖️ Dataset & Domain Comparison

| Dimension | `FRANCE_DATASET` | `INDIAN_DATASET` |
|---|---|---|
| **Forecasting Scope** | Single Household (Micro-level) | National Electric Grid (Macro-level) |
| **Location** | Sceaux (near Paris), France | All-India National Grid Monitors |
| **Sampling Resolution** | 1-minute intervals | 1-hour intervals |
| **Target Variable** | `Global_active_power` (Kilowatts - kW) | `National Hourly Demand` (Megawatts - MW) |
| **Primary Features** | Autoregressive lag features (1, 5, 15, 60, 1440 mins) + calendar timestamps | Calendar timestamps (`hour`, `day`, `day_of_week`, `month`, `year`, `day_of_year`) + sequence windows |
| **Data Format** | Semicolon-delimited text (`.txt`, ~133 MB, 2,075,259 records) | Microsoft Excel spreadsheet (`.xlsx`, ~3.3 MB) |
| **Notebook Architecture** | **Decoupled**: 5 dedicated training notebooks + 5 dedicated Gradio deployment notebooks | **Unified**: 5 all-in-one notebooks containing data prep, training, metrics, and Gradio interface |
| **Best Performing Model** | XGBoost / Random Forest ($R^2 \approx 0.9418$, $\text{RMSE} \approx 0.216$ kW) | LSTM Deep Learning ($R^2 \approx 0.9779$, $\text{RMSE} \approx 2,904$ MW) |

---

## 🇫🇷 Folder 1: FRANCE_DATASET (Household Power Consumption)

### France Overview & Raw Data
- **File**: `FRANCE_DATASET/household_power_consumption.csv`
- **Sample Rate**: 1 minute between December 2006 and November 2010 (~47 months).
- **Size**: 2,075,259 rows, 9 attributes.
- **Attributes**:
  - `Date`: Date formatted as `dd/mm/yyyy`.
  - `Time`: Time formatted as `hh:mm:ss`.
  - `Global_active_power`: Household global minute-averaged active power (in kW, target).
  - `Global_reactive_power`: Household global minute-averaged reactive power (in kW).
  - `Voltage`: Minute-averaged voltage (in Volts).
  - `Global_intensity`: Minute-averaged current intensity (in Amperes).
  - `Sub_metering_1`: Energy sub-metering No. 1 (kitchen: dishwasher, microwave).
  - `Sub_metering_2`: Energy sub-metering No. 2 (laundry room: washing machine, dryer, light).
  - `Sub_metering_3`: Energy sub-metering No. 3 (electric water heater and air conditioner).

### France Notebook Architecture
The folder provides two sets of notebooks to cleanly separate training experimentation from production UI deployment:

1. **Standalone Training Notebooks**:
   - [Linear Regression.ipynb](file:///d:/Documents%20(D)/PBL_Model/FRANCE_DATASET/Linear%20Regression.ipynb): Trains baseline OLS regression using physical electrical measurements (`Voltage`, `Global_reactive_power`, `Global_intensity`, sub-meterings).
   - [LSTM.ipynb](file:///d:/Documents%20(D)/PBL_Model/FRANCE_DATASET/LSTM.ipynb): Sequence-based LSTM trained on 30-step sliding windows of active power.
   - [RFR.ipynb](file:///d:/Documents%20(D)/PBL_Model/FRANCE_DATASET/RFR.ipynb): Random Forest Regressor trained on 30-step sliding sequence representations.
   - [SVR.ipynb](file:///d:/Documents%20(D)/PBL_Model/FRANCE_DATASET/SVR.ipynb): Support Vector Regressor with RBF kernel trained on 30-step sequence representations.
   - [XGBoost.ipynb](file:///d:/Documents%20(D)/PBL_Model/FRANCE_DATASET/XGBoost.ipynb): Extreme Gradient Boosting Regressor trained on 30-step sequence representations.

2. **Gradio Deployment Notebooks**:
   - [LR_Gradio.ipynb](file:///d:/Documents%20(D)/PBL_Model/FRANCE_DATASET/LR_Gradio.ipynb)
   - [LSTM_Gradio.ipynb](file:///d:/Documents%20(D)/PBL_Model/FRANCE_DATASET/LSTM_Gradio.ipynb)
   - [RFR_Gradio.ipynb](file:///d:/Documents%20(D)/PBL_Model/FRANCE_DATASET/RFR_Gradio.ipynb)
   - [SVR_Gradio.ipynb](file:///d:/Documents%20(D)/PBL_Model/FRANCE_DATASET/SVR_Gradio.ipynb)
   - [XGBoost_Gradio.ipynb](file:///d:/Documents%20(D)/PBL_Model/FRANCE_DATASET/XGBoost_Gradio.ipynb)

### France Feature Engineering Strategies
1. **Autoregressive Lag Features** (used in Gradio models):
   - `Power_Lag_1`: Active power 1 minute prior.
   - `Power_Lag_5`: Active power 5 minutes prior.
   - `Power_Lag_15`: Active power 15 minutes prior.
   - `Power_Lag_60`: Active power 1 hour prior.
   - `Power_Lag_1440`: Active power 24 hours prior (daily cycle).
2. **Temporal Features**:
   - `Year`, `Month` (1–12), `Day` (1–31), `Hour` (0–23), `Minute` (0–59), `Day_of_Week` (0=Monday to 6=Sunday).
3. **Sequential Windowing** (used in standalone sequence notebooks):
   - 30-minute historical windows (`time_steps = 30`) scaled via `MinMaxScaler(feature_range=(0, 1))`.

### France Model Performance Benchmark

| Model | Evaluation Framework | MAE (kW) | RMSE (kW) | $R^2$ Score | Key Observations |
|---|---|---|---|---|---|
| **XGBoost** | Lag Features (Gradio) | **0.0821** | **0.2161** | **0.9418** | Best overall performance; fast execution |
| **Random Forest** | Lag Features (Gradio) | 0.0829 | 0.2168 | 0.9414 | Near identical accuracy to XGBoost |
| **Linear Regression** | Lag Features (Gradio) | 0.0850 | 0.2204 | 0.9395 | Extremely strong baseline with lag inputs |
| **LSTM** | 60-step Window (Gradio) | 0.1145 | 0.2293 | 0.9345 | Strong sequence capture, handles non-linear patterns |
| **SVR** | Lag Features (Gradio) | 0.2190 | 0.3554 | 0.8427 | Good accuracy, higher compute cost for large sets |
| **Random Forest** | 30-step Window | 0.1668 | 0.3796 | 0.9354 | Effective on sliding sequences |
| **XGBoost** | 30-step Window | 0.1765 | 0.3823 | 0.9345 | Fast convergence on sequence matrices |
| **LSTM** | 30-step Window | 0.2577 | 0.4783 | 0.8714 | Effective sequence learning |
| **SVR** | 30-step Window | 0.2620 | 0.5781 | 0.8502 | Captures RBF non-linear boundaries |

*(Note: In [Linear Regression.ipynb](file:///d:/Documents%20(D)/PBL_Model/FRANCE_DATASET/Linear%20Regression.ipynb), using direct concurrent physical features like `Global_intensity` yields an $R^2$ of 0.9985 due to direct electrical Ohm's law correlation $P \approx V \cdot I$).*

### France Gradio Interactive Interfaces
- **Tabular Models (LR, RFR, SVR, XGBoost)**:
  - **Inputs**:
    - 5 numeric inputs: `power_lag_1`, `power_lag_5`, `power_lag_15`, `power_lag_60`, `power_lag_1440`.
    - 6 sliders: `Year` (numeric), `Month` (1–12), `Day` (1–31), `Hour` (0–23), `Minute` (0–59), `Day of Week` (0–6).
  - **Output**: Formatted text displaying `Predicted Global Active Power: X.XXXX kW`.
- **LSTM Model**:
  - **Input**: Integer dataset index `index` ($\ge 60$).
  - **Output**: Formatted prediction showing actual historical sequence versus predicted active power.

### France Saved Artifacts & Outputs
- `FRANCE_DATASET/Linear_Regression_Output/`:
  - `linear_regression_energy_model.pkl`: Serialized Linear Regression model.
- `FRANCE_DATASET/LSTM_Output/`:
  - `linear_regression_energy_model.pkl`
  - `smart_grid_lstm_model.keras`: Saved Keras Sequential LSTM model.
  - `lstm_predictions.csv`: Model test predictions.
- `FRANCE_DATASET/RFR_Outputs/`:
  - `random_forest_model.pkl`: Saved Scikit-Learn `RandomForestRegressor`.
  - `random_forest_predictions.csv`: Test set predictions.
- `FRANCE_DATASET/SVR_Output/`:
  - `smart_grid_svr_model.pkl`: Saved Scikit-Learn `SVR`.
  - `svr_predictions.csv`: Test set predictions.
- `FRANCE_DATASET/XGBoost_Output/`:
  - `xgboost_model.pkl`: Saved XGBoost Regressor model.
  - `xgboost_predictions.csv`: Test set predictions.

---

## 🇮🇳 Folder 2: INDIAN_DATASET (National Electricity Demand)

### India Overview & Raw Data
- **Directory**: `INDIAN_DATASET/dataset/`
- **Primary File**: `hourlyLoadDataIndia.xlsx` (Hourly national load monitor records, measured in Megawatts - MW).
- **Auxiliary File**: `monthly_temp.xlsx` (Monthly temperature records across Indian meteorological regions).
- **Target Variable**: `National Hourly Demand` (MW).

### India Notebook Architecture
All notebooks in `INDIAN_DATASET` are unified, end-to-end notebooks containing data loading, preprocessing, model training, evaluation metrics, and the Gradio web interface:

1. [LR_Gradio.ipynb](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/LR_Gradio.ipynb): Ordinary Least Squares regression using temporal features.
2. [LSTM_Gradio.ipynb](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/LSTM_Gradio.ipynb): 2-Layer LSTM with Dropout (`64` units $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `32` units $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(16)` $\rightarrow$ `Dense(1)`).
3. [RFR_Gradio.ipynb](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/RFR_Gradio.ipynb): Random Forest Regressor (`n_estimators=100`, `max_depth=20`, `n_jobs=-1`).
4. [SVR_Gradio.ipynb](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/SVR_Gradio.ipynb): Support Vector Regressor with RBF kernel and `StandardScaler` feature normalization.
5. [XGBoost_Gradio.ipynb](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/XGBoost_Gradio.ipynb): XGBoost Regressor (`n_estimators=300`, `learning_rate=0.05`, `max_depth=8`).

### India Feature Engineering Strategies
1. **Timestamp Decomposition**:
   - `hour` (0–23)
   - `day` (1–31)
   - `day_of_week` (0=Monday to 6=Sunday)
   - `month` (1–12)
   - `year`
   - `day_of_year` (1–366)
2. **Sequential Load Histories (LSTM)**:
   - Rolling lookback sequence of 24 hourly steps (`TIME_STEPS = 24`) scaled with `MinMaxScaler`.
   - Captures intra-day diurnal curves and consecutive-day load shifts.

### India Model Performance Benchmark

| Model | Feature Type | MAE (MW) | RMSE (MW) | $R^2$ Score | Key Insights |
|---|---|---|---|---|---|
| **LSTM Deep Neural Network** | 24-step Sequence Windows | **2,245.32** | **2,904.14** | **+0.9779** | **Superior model.** Captures true sequential dispatch and grid inertia |
| **Random Forest Regressor** | Calendar Only | 16,030.64 | 19,612.99 | -0.0093 | Static calendar features alone cannot resolve rapid load shifts |
| **XGBoost Regressor** | Calendar Only | 16,304.42 | 19,745.77 | -0.0230 | Captures basic non-linear trends but lacks load autoregression |
| **SVR (RBF Kernel)** | Standardized Calendar | 16,883.06 | 20,446.05 | -0.0968 | Smooth boundary mapping; limited by static calendar data |
| **Linear Regression** | Calendar Only | 17,303.20 | 20,854.42 | -0.1411 | Linear fit insufficient for multi-seasonal national variations |

> [!TIP]
> **Key Domain Finding**: On national electricity demand, deep sequential models (**LSTM**) outperform static calendar regressions by orders of magnitude ($R^2 = 0.9779$ vs negative $R^2$ on calendar-only features), demonstrating that electricity demand requires autoregressive sequence history or temperature covariates rather than pure timestamp labels.

### India Gradio Interactive Interfaces
All 5 notebooks in `INDIAN_DATASET` share an intuitive, unified interface:
- **Inputs**:
  - `Date`: Textbox expecting `YYYY-MM-DD` (e.g., `2024-05-15`).
  - `Time`: Textbox expecting `HH:MM` or `HH:MM:SS` (e.g., `14:30`).
- **Inference Logic**:
  - Automatically parses the date and time strings.
  - Extracts calendar features (`hour`, `day`, `day_of_week`, `month`, `year`, `day_of_year`).
  - For LSTM: locates the nearest historical timestamps to build the sequential tensor.
- **Output**:
  - Formatted card displaying predicted demand in Megawatts (MW) and inference status.

### India Saved Artifacts & Outputs
- `INDIAN_DATASET/Linear Regression Model Data/linear_regression_model.pkl`: Serialized Scikit-Learn linear regression model.
- `INDIAN_DATASET/RFR Model Data/random_forest_model.pkl`: Serialized `RandomForestRegressor` (~302 MB).
- `INDIAN_DATASET/SVR Model Data/`:
  - `svr_model.pkl`: Serialized Support Vector Regressor (~2.5 MB).
  - `svr_scaler.pkl`: Serialized `StandardScaler` fitted on training features.

---

## 🤖 Machine Learning & Deep Learning Architectures

### 1. Linear Regression (Baseline)
- Ordinary Least Squares (OLS) implemented via `scikit-learn.linear_model.LinearRegression`.
- Serves as the baseline model for both domains.

### 2. Random Forest Regressor
- Non-linear ensemble model implemented via `sklearn.ensemble.RandomForestRegressor`.
- Configured with `n_estimators=100`, `max_depth=20`, and `n_jobs=-1` for multi-core parallelism.

### 3. Support Vector Regression (SVR)
- Kernel regression implemented via `sklearn.svm.SVR`.
- Uses Radial Basis Function (`kernel='rbf'`, $C=100$, $\gamma=\text{'scale'}$, $\epsilon=0.1$).
- Paired with `StandardScaler` or `MinMaxScaler` for numerical stability.

### 4. XGBoost Regressor
- Gradient Boosted Decision Trees implemented via `xgboost.XGBRegressor`.
- Configured with `n_estimators=300`, `learning_rate=0.05`, and `max_depth=8`.

### 5. LSTM (Long Short-Term Memory) Deep Neural Networks
- Recurrent neural networks built using `tensorflow.keras.models.Sequential`:
  - **France Dataset LSTM**: `LSTM(50)` + `Dropout(0.2)` + `Dense(25, relu)` + `Dense(1)` (or `LSTM(32)` + `Dropout(0.1)` + `Dense(16)` + `Dense(1)`).
  - **India Dataset LSTM**: `LSTM(64, return_sequences=True)` + `Dropout(0.2)` + `LSTM(32)` + `Dropout(0.2)` + `Dense(16, relu)` + `Dense(1)`.
  - Loss function: Mean Squared Error (`mse`).
  - Optimizer: Adam.

---

## ⚙️ Environment Setup & Installation

### Python Version
- **Recommended Python**: **Python 3.10 to 3.12** (Python 3.13/3.14 with compatible pre-built wheels also supported).
- All libraries have been tested for local CPU execution and optional GPU acceleration.

### Virtual Environment Setup

#### Windows (PowerShell)
```powershell
# Navigate to the project root
cd "d:\Documents (D)\PBL_Model"

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt)
```cmd
cd "d:\Documents (D)\PBL_Model"
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS / Linux
```bash
cd /path/to/PBL_Model
python3 -m venv .venv
source .venv/bin/activate
```

### Installing Requirements
Upgrade `pip` and install all dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Contents of [requirements.txt](file:///d:/Documents%20(D)/PBL_Model/requirements.txt):
```text
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
joblib>=1.3.0
openpyxl>=3.1.0
xgboost>=2.0.0
tensorflow>=2.15.0
gradio>=4.0.0
jupyter>=1.0.0
ipykernel>=6.20.0
```

---

## 💻 VS Code & Jupyter Setup

1. Open the project root in VS Code:
   ```bash
   code .
   ```
2. Ensure the **Python** and **Jupyter** extensions are installed in VS Code.
3. Open any notebook from [FRANCE_DATASET/](file:///d:/Documents%20(D)/PBL_Model/FRANCE_DATASET) or [INDIAN_DATASET/](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET).
4. Click **Select Kernel** in the upper right corner of the notebook editor.
5. Choose **Python Environments...** and select `.venv` (or your Python environment).
6. Click **Run All** or execute cells individually.

---

## 🚀 Execution (CPU & GPU)

### CPU Execution
- Every notebook runs out of the box on standard laptop/desktop CPUs without requiring CUDA or discrete GPUs.
- Multi-core processing is enabled by default with `n_jobs=-1` in Scikit-Learn and XGBoost.

### GPU Acceleration
- If an NVIDIA GPU with compatible CUDA/cuDNN drivers is available, TensorFlow automatically leverages GPU acceleration.
- Safe dynamic memory allocation is embedded in all TensorFlow notebooks:
  ```python
  import tensorflow as tf
  gpus = tf.config.list_physical_devices("GPU")
  if gpus:
      for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
  ```
- On Windows, native TensorFlow $\ge 2.11$ defaults to oneDNN CPU optimizations, which execute quickly for these sequence sizes.

---

## 🌐 Running Gradio Web Applications

Each Gradio notebook launches a local web server:

1. Open any `*_Gradio.ipynb` notebook:
   - **France Models**: `LR_Gradio.ipynb`, `LSTM_Gradio.ipynb`, `RFR_Gradio.ipynb`, `SVR_Gradio.ipynb`, `XGBoost_Gradio.ipynb`
   - **India Models**: `LR_Gradio.ipynb`, `LSTM_Gradio.ipynb`, `RFR_Gradio.ipynb`, `SVR_Gradio.ipynb`, `XGBoost_Gradio.ipynb`
2. Run the notebook through the final cell (`demo.launch()` or `interface.launch()`).
3. Open the displayed local URL in your web browser:
   ```
   Running on local URL:  http://127.0.0.1:7860
   ```
4. Enter input values and click **Submit** to view the predicted load instantly.
5. All Gradio interfaces run in local-only mode (`share=False`) with zero external network transmission.

---

## 🔧 Troubleshooting & FAQs

| Issue | Root Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'openpyxl'` | Excel engine missing | Run `pip install openpyxl` in your active virtual environment. |
| `InconsistentVersionWarning` when unpickling `.pkl` | Model serialized with earlier `scikit-learn` version | This is a non-fatal warning. If desired, re-run the training notebook to overwrite the `.pkl` using your current version. |
| Gradio port already in use (`Address already in use: 7860`) | An earlier notebook kernel is still hosting Gradio | Restart the previous notebook's kernel or specify a custom port: `demo.launch(server_port=7861)`. |
| Out of memory during LSTM training on France dataset | Full ~2.07M rows processed at once | Reduce the subset sample size (e.g., first 50,000–100,000 rows) or decrease `batch_size` (e.g., from 256 to 64 or 32). |
| Relative path errors when loading data files | Jupyter working directory differs from script root | All notebooks in this project include resilient `Path` resolution that checks both the current directory and the parent directory automatically. |
