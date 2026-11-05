# Electricity Demand & Load Forecasting (PBL Machine Learning Project)

A portable, cross-platform machine learning and deep learning project for forecasting electrical energy demand across household (micro-scale) and national grid (macro-scale) domains.

> [!NOTE]
> **This project runs completely offline and locally.** All raw datasets, pre-trained model artifacts, feature engineering pipelines, training workflows, and Gradio web applications run on your local machine using standard Python and VS Code — no Google Colab, Google Drive, or cloud subscriptions required.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Source Files & Repository Inventory](#-source-files--repository-inventory)
  - [Directory Tree](#directory-tree)
  - [Detailed File Breakdown](#detailed-file-breakdown)
- [Dataset & Domain Comparison](#-dataset--domain-comparison)
- [Essential Tools, Technologies & Software Stack](#-essential-tools-technologies--software-stack)
  - [Core Programming Language](#core-programming-language)
  - [Data Manipulation & Numerical Processing](#data-manipulation--numerical-processing)
  - [Machine Learning Algorithms](#machine-learning-algorithms)
  - [Deep Learning Architecture](#deep-learning-architecture)
  - [Feature Scaling & Preprocessing](#feature-scaling--preprocessing)
  - [Interactive Web UI Framework](#interactive-web-ui-framework)
  - [Model Persistence & Serialization](#model-persistence--serialization)
  - [Data Visualization](#data-visualization)
  - [Development Environment & Extensions](#development-environment--extensions)
  - [Version Control & Large File Storage](#version-control--large-file-storage)
- [Machine Learning & Deep Learning Architectures](#-machine-learning--deep-learning-architectures)
  - [1. Ordinary Least Squares Linear Regression](#1-ordinary-least-squares-linear-regression)
  - [2. Random Forest Regressor](#2-random-forest-regressor)
  - [3. Support Vector Regression (SVR)](#3-support-vector-regression-svr)
  - [4. XGBoost Regressor](#4-xgboost-regressor)
  - [5. Long Short-Term Memory (LSTM) Networks](#5-long-short-term-memory-lstm-networks)
- [Model Performance Benchmark & Insights](#-model-performance-benchmark--insights)
  - [Foreign Dataset (Household Level)](#foreign-dataset-household-level)
  - [Indian Dataset (National Grid Level)](#indian-dataset-national-grid-level)
- [Step-by-Step Execution & Usage Guide](#-step-by-step-execution--usage-guide)
  - [Prerequisites](#prerequisites)
  - [Step 1: Clone or Navigate to the Repository](#step-1-clone-or-navigate-to-the-repository)
  - [Step 2: Create and Activate a Virtual Environment](#step-2-create-and-activate-a-virtual-environment)
  - [Step 3: Install Required Dependencies](#step-3-install-required-dependencies)
  - [Step 4: Verify Git LFS Assets (Optional but Recommended)](#step-4-verify-git-lfs-assets-optional-but-recommended)
  - [Step 5: Configure VS Code and Jupyter Kernel](#step-5-configure-vs-code-and-jupyter-kernel)
  - [Step 6: Executing Training Notebooks](#step-6-executing-training-notebooks)
  - [Step 7: Launching and Using the Gradio Web Applications](#step-7-launching-and-using-the-gradio-web-applications)
  - [Step 8: Standalone Python Inference (Direct Model Loading)](#step-8-standalone-python-inference-direct-model-loading)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🔍 Project Overview

Accurate electrical load forecasting is essential for modern power systems: it prevents grid overloads, informs optimal dispatch scheduling, reduces costly thermal spinning reserves, facilitates renewable integration, and enables consumers to manage peak electricity costs.

This project delivers an end-to-end comparative study across two complementary scales of energy systems:

1. **Micro-Scale Household Power Forecasting (`FOREIGN_DATASET`)**:
   - Forecasts minute-by-minute active electrical power (`Global_active_power` in kW) consumed by a residential home in Sceaux, France.
   - Evaluates autoregressive lag features ($t-1$, $t-5$, $t-15$, $t-60$, $t-1440$ minutes), temporal calendar cyclical signals, and sliding multi-step sequence windows.
2. **Macro-Scale National Grid Load Forecasting (`INDIAN_DATASET`)**:
   - Forecasts hourly electricity demand (`National Hourly Demand` in MW) across the Indian national power grid.
   - Benchmarks static calendar-based regression against deep sequential recurrent modeling (LSTM), demonstrating how sequential temporal inertia dominates macro-scale grid dispatch.

Every model in both domains is equipped with an interactive, browser-based **Gradio** web interface for real-time inference and evaluation.

---

## 📁 Source Files & Repository Inventory

### Directory Tree

```
PBL_Model/
├── .gitattributes                                       # Git Large File Storage (LFS) tracking configuration
├── .gitignore                                           # Excluded build artifacts, environments, and caches
├── requirements.txt                                     # Unified Python dependencies
├── README.md                                            # Comprehensive project documentation
│
├── FOREIGN_DATASET/                                     # Household-level micro power forecasting
│   ├── household_power_consumption.csv                  # Raw dataset (~2.075M records, ~133 MB)
│   │
│   ├── Linear Regression.ipynb                          # OLS regression training (physical & calendar features)
│   ├── LR_Gradio.ipynb                                  # OLS regression training + Gradio web interface
│   ├── LSTM.ipynb                                       # LSTM sequential training (30-step sliding window)
│   ├── LSTM_Gradio.ipynb                                # LSTM sequential training + Gradio interactive UI
│   ├── RFR.ipynb                                        # Random Forest training (30-step sliding window)
│   ├── RFR_Gradio.ipynb                                 # Random Forest training + Gradio web interface
│   ├── SVR.ipynb                                        # Support Vector Regressor training (30-step window)
│   ├── SVR_Gradio.ipynb                                 # SVR training + Gradio web interface
│   ├── XGBoost.ipynb                                    # XGBoost Regressor training (30-step window)
│   ├── XGBoost_Gradio.ipynb                             # XGBoost training + Gradio web interface
│   │
│   ├── Linear_Regression_Output/
│   │   └── linear_regression_energy_model.pkl           # Pre-trained Scikit-Learn Linear Regression model
│   ├── LSTM_Output/
│   │   ├── linear_regression_energy_model.pkl           # Saved baseline model artifact
│   │   ├── smart_grid_lstm_model.keras                  # Pre-trained Keras LSTM neural network (~90 KB)
│   │   └── lstm_predictions.csv                         # Test partition actuals vs. model predictions
│   ├── RFR_Outputs/
│   │   ├── random_forest_model.pkl                      # Pre-trained Scikit-Learn RandomForest (~18 MB)
│   │   └── random_forest_predictions.csv                # Test partition actuals vs. model predictions
│   ├── SVR_Output/
│   │   ├── smart_grid_svr_model.pkl                     # Pre-trained Scikit-Learn SVR model (~1.8 MB)
│   │   └── svr_predictions.csv                          # Test partition actuals vs. model predictions
│   └── XGBoost_Output/
│       ├── xgboost_model.pkl                            # Pre-trained XGBoost Regressor model (~282 KB)
│       └── xgboost_predictions.csv                      # Test partition actuals vs. model predictions
│
└── INDIAN_DATASET/                                      # National grid macro electricity demand forecasting
    ├── dataset/
    │   ├── hourlyLoadDataIndia.xlsx                     # Historical hourly national demand (~3.3 MB)
    │   └── monthly_temp.xlsx                            # Meteorological monthly temperature records (~11 KB)
    │
    ├── LR_Gradio.ipynb                                  # End-to-end Linear Regression pipeline & Gradio UI
    ├── LSTM_Gradio.ipynb                                # End-to-end 2-Layer LSTM deep neural net & Gradio UI
    ├── RFR_Gradio.ipynb                                 # End-to-end Random Forest Regressor & Gradio UI
    ├── SVR_Gradio.ipynb                                 # End-to-end SVR pipeline (with StandardScaler) & UI
    ├── XGBoost_Gradio.ipynb                             # End-to-end XGBoost Regressor & Gradio UI
    │
    ├── Linear Regression Model Data/
    │   └── linear_regression_model.pkl                  # Pre-trained Scikit-Learn Linear Regression model
    ├── LSTM Model Data/
    │   └── lstm_model.keras                             # Pre-trained Keras 2-Layer LSTM network (~400 KB)
    ├── RFR Model Data/
    │   └── random_forest_model.pkl                      # Pre-trained Scikit-Learn RandomForest (~302 MB)
    └── SVR Model Data/
        ├── svr_model.pkl                                # Pre-trained Scikit-Learn Support Vector Regressor
        └── svr_scaler.pkl                               # Pre-trained StandardScaler for input normalization
```

---

### Detailed File Breakdown

#### 1. Root Configuration & Documentation Files
- [`.gitattributes`](file:///d:/Documents%20(D)/PBL_Model/.gitattributes): Configures Git Large File Storage (LFS) filters for model weights (`*.keras`, `*.pkl`) and tabular datasets (`*.csv`, `*.txt`), preventing repository bloat and corruption during versioning.
- [`.gitignore`](file:///d:/Documents%20(D)/PBL_Model/.gitignore): Standard rules omitting Python virtual environments (`.venv/`), bytecode caches (`__pycache__/`), and Jupyter checkpoints (`.ipynb_checkpoints/`).
- [`requirements.txt`](file:///d:/Documents%20(D)/PBL_Model/requirements.txt): Specification of required runtime packages with minimum version constraints.
- [`README.md`](file:///d:/Documents%20(D)/PBL_Model/README.md): Primary architecture guide, source file index, technology inventory, and execution manual.

#### 2. Raw Datasets
- [`FOREIGN_DATASET/household_power_consumption.csv`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/household_power_consumption.csv):
  - **Volume**: 2,075,259 measurements collected between December 2006 and November 2010.
  - **Columns**: `Date` (dd/mm/yyyy), `Time` (hh:mm:ss), `Global_active_power` (kW), `Global_reactive_power` (kW), `Voltage` (V), `Global_intensity` (A), `Sub_metering_1` (kitchen: dishwasher/microwave in Wh), `Sub_metering_2` (laundry: washing machine/dryer in Wh), `Sub_metering_3` (climate control: electric water-heater and air-conditioner in Wh).
- [`INDIAN_DATASET/dataset/hourlyLoadDataIndia.xlsx`](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/dataset/hourlyLoadDataIndia.xlsx):
  - **Volume**: Multi-year hourly time-series tracking total power dispatch across the Indian national power grid.
  - **Key Variables**: Timestamp indices and hourly aggregated load demand in Megawatts (MW).
- [`INDIAN_DATASET/dataset/monthly_temp.xlsx`](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/dataset/monthly_temp.xlsx):
  - **Volume**: Regional monthly mean temperature records across India, providing seasonal meteorological context.

#### 3. Foreign Dataset Source Notebooks
The Foreign dataset implements a decoupled structure with separate training-only and Gradio deployment notebooks:
- **Offline Training Notebooks**:
  - [`Linear Regression.ipynb`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/Linear%20Regression.ipynb): Demonstrates OLS regression on physical circuit measurements and temporal tags.
  - [`LSTM.ipynb`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/LSTM.ipynb): Implements a sequential LSTM network on 30-minute sliding sequence matrices.
  - [`RFR.ipynb`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/RFR.ipynb): Trains a multi-core `RandomForestRegressor` on flattened 30-step temporal matrices.
  - [`SVR.ipynb`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/SVR.ipynb): Explores Support Vector Regression using an RBF kernel on scaled sequential data.
  - [`XGBoost.ipynb`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/XGBoost.ipynb): Implements gradient boosting on sliding 30-step window vectors.
- **Interactive Gradio Deployment Notebooks**:
  - [`LR_Gradio.ipynb`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/LR_Gradio.ipynb): Trains an OLS model on autoregressive lag features ($t-1, t-5, t-15, t-60, t-1440$) and serves a Gradio web UI.
  - [`LSTM_Gradio.ipynb`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/LSTM_Gradio.ipynb): Builds a 60-step lookback LSTM sequence model and launches a web UI for index-based evaluation.
  - [`RFR_Gradio.ipynb`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/RFR_Gradio.ipynb): Trains a Random Forest regressor with lag features and provides a web UI.
  - [`SVR_Gradio.ipynb`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/SVR_Gradio.ipynb): Fits an RBF-kernel SVR on normalized lag features with an integrated UI.
  - [`XGBoost_Gradio.ipynb`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/XGBoost_Gradio.ipynb): Fits an `XGBRegressor` on lag and calendar features, achieving the highest household-level accuracy ($R^2 = 0.9418$), integrated with Gradio.

#### 4. Indian Dataset Source Notebooks
The Indian dataset notebooks are unified, end-to-end notebooks containing data loading, calendar feature engineering, model training, evaluation, and the Gradio user interface:
- [`LR_Gradio.ipynb`](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/LR_Gradio.ipynb): OLS regression on decomposed calendar timestamp features.
- [`LSTM_Gradio.ipynb`](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/LSTM_Gradio.ipynb): 2-Layer recurrent LSTM network (`64` units $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `32` units $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(16)` $\rightarrow$ `Dense(1)`) trained over 24-hour historical load windows ($R^2 = 0.9779$).
- [`RFR_Gradio.ipynb`](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/RFR_Gradio.ipynb): Non-linear ensemble Random Forest on calendar features with `n_estimators=100`.
- [`SVR_Gradio.ipynb`](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/SVR_Gradio.ipynb): Support Vector Regressor equipped with `StandardScaler` feature scaling.
- [`XGBoost_Gradio.ipynb`](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/XGBoost_Gradio.ipynb): Gradient-boosted decision trees (`learning_rate=0.05`, `max_depth=8`, `n_estimators=300`).

#### 5. Pre-Trained Model Artifacts & Output Exports
- **Foreign Dataset Artifacts**:
  - [`FOREIGN_DATASET/Linear_Regression_Output/linear_regression_energy_model.pkl`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/Linear_Regression_Output/linear_regression_energy_model.pkl): Pickled baseline linear model.
  - [`FOREIGN_DATASET/LSTM_Output/smart_grid_lstm_model.keras`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/LSTM_Output/smart_grid_lstm_model.keras): Native Keras format LSTM network.
  - [`FOREIGN_DATASET/LSTM_Output/lstm_predictions.csv`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/LSTM_Output/lstm_predictions.csv): Test set actual vs. predicted values.
  - [`FOREIGN_DATASET/RFR_Outputs/random_forest_model.pkl`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/RFR_Outputs/random_forest_model.pkl): Serialized Random Forest model (~18 MB).
  - [`FOREIGN_DATASET/RFR_Outputs/random_forest_predictions.csv`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/RFR_Outputs/random_forest_predictions.csv): Random Forest test predictions.
  - [`FOREIGN_DATASET/SVR_Output/smart_grid_svr_model.pkl`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/SVR_Output/smart_grid_svr_model.pkl): Serialized SVR model (~1.8 MB).
  - [`FOREIGN_DATASET/SVR_Output/svr_predictions.csv`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/SVR_Output/svr_predictions.csv): SVR test predictions.
  - [`FOREIGN_DATASET/XGBoost_Output/xgboost_model.pkl`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/XGBoost_Output/xgboost_model.pkl): Serialized XGBoost model (~282 KB).
  - [`FOREIGN_DATASET/XGBoost_Output/xgboost_predictions.csv`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/XGBoost_Output/xgboost_predictions.csv): XGBoost test predictions.
- **Indian Dataset Artifacts**:
  - [`INDIAN_DATASET/Linear Regression Model Data/linear_regression_model.pkl`](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/Linear%20Regression%20Model%20Data/linear_regression_model.pkl): Pickled OLS model.
  - [`INDIAN_DATASET/LSTM Model Data/lstm_model.keras`](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/LSTM%20Model%20Data/lstm_model.keras): Serialized 2-Layer LSTM network (~400 KB).
  - [`INDIAN_DATASET/RFR Model Data/random_forest_model.pkl`](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/RFR%20Model%20Data/random_forest_model.pkl): Pickled Random Forest model (~302 MB).
  - [`INDIAN_DATASET/SVR Model Data/svr_model.pkl`](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/SVR%20Model%20Data/svr_model.pkl): Pickled SVR model (~2.5 MB).
  - [`INDIAN_DATASET/SVR Model Data/svr_scaler.pkl`](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/SVR%20Model%20Data/svr_scaler.pkl): Fitted `StandardScaler` used for normalizing inference features.

---

## ⚖️ Dataset & Domain Comparison

| Dimension | `FOREIGN_DATASET` (Household Level) | `INDIAN_DATASET` (National Grid Level) |
|---|---|---|
| **Forecasting Scope** | Single Household (Micro-level) | National Electric Grid (Macro-level) |
| **Location** | Sceaux (near Paris), France | All-India National Grid Monitors |
| **Temporal Resolution** | 1-minute intervals | 1-hour intervals |
| **Target Metric** | `Global_active_power` (Kilowatts — kW) | `National Hourly Demand` (Megawatts — MW) |
| **Primary Features** | Autoregressive lag features (1, 5, 15, 60, 1440 mins) + calendar timestamps | Calendar timestamps (`hour`, `day`, `day_of_week`, `month`, `year`, `day_of_year`) + 24-step sequence windows |
| **Raw Data File** | `household_power_consumption.csv` (~133 MB, 2.075M records) | `hourlyLoadDataIndia.xlsx` (~3.3 MB, Excel format) |
| **Notebook Architecture** | **Decoupled**: 5 training-only notebooks + 5 Gradio UI notebooks | **Unified**: 5 all-in-one notebooks (loading, training, evaluation, Gradio UI) |
| **Top Performing Model** | **XGBoost / Random Forest** ($R^2 \approx 0.9418$, $\text{RMSE} \approx 0.216$ kW) | **LSTM Deep Neural Network** ($R^2 = 0.9779$, $\text{RMSE} = 2,904$ MW) |
| **Key Physical Insight** | Short-term lags ($t-1$, $t-5$) provide high predictive power for individual appliance duty cycles | National grid load exhibits strong continuous inertia requiring 24h sequential state memory |

---

## 🛠️ Essential Tools, Technologies & Software Stack

This project is built using modern, production-grade open-source tools across the Python data science and machine learning ecosystem:

### Core Programming Language
- **Python (v3.10 to v3.12 recommended)**:
  - Serves as the primary runtime language.
  - Utilizes standard library modules (`os`, `sys`, `pathlib.Path`, `datetime`, `pickle`, `math`, `warnings`).
  - Cross-platform compatible across Windows (x64), Linux (x86_64/ARM64), and macOS (Apple Silicon/Intel).

### Data Manipulation & Numerical Processing
- **Pandas (`>= 2.0.0`)**:
  - Ingests large tabular files (`read_csv` with custom delimiters and `read_excel`).
  - Time-series handling: string parsing to `pd.to_datetime`, timestamp indexing, datetime attribute extraction (`.dt.hour`, `.dt.dayofweek`, `.dt.dayofyear`), and rolling windows.
  - Handles missing value imputation (`fillna`, forward/backward fill).
- **NumPy (`>= 1.24.0`)**:
  - Fast multidimensional vector and matrix calculations.
  - Constructing 3D input tensors (`[samples, time_steps, features]`) for recurrent neural network consumption.
- **OpenPyXL (`>= 3.1.0`)**:
  - Underlying engine enabling Pandas to parse Microsoft Excel `.xlsx` spreadsheets for the Indian dataset without requiring Microsoft Office.

### Machine Learning Algorithms
- **Scikit-Learn (`>= 1.3.0`)**:
  - `LinearRegression`: Ordinary Least Squares linear regression used as the foundational baseline.
  - `RandomForestRegressor`: Ensemble bagged decision trees operating with multi-threaded parallel trees (`n_jobs=-1`).
  - `SVR`: Support Vector Regression mapping non-linear continuous relationships via Radial Basis Function (RBF) kernels.
  - `train_test_split`: Chronological dataset splitting (`shuffle=False` for time series).
  - `metrics`: Evaluation functions (`mean_squared_error`, `mean_absolute_error`, `r2_score`).
- **XGBoost (`>= 2.0.0`)**:
  - High-performance gradient boosted decision trees (`XGBRegressor`).
  - Optimized tree pruning, regularization ($\text{L1}/\text{L2}$ penalties), and multi-core CPU execution.

### Deep Learning Architecture
- **TensorFlow & Keras (`>= 2.15.0`)**:
  - Recurrent neural networks utilizing the `Sequential` model architecture.
  - Core layers: `LSTM` (recurrent hidden state cells), `Dropout` (stochastic regularization to prevent overfitting), `Dense` (linear and ReLU decision heads).
  - Training components: `Adam` adaptive gradient descent optimizer, Mean Squared Error (`mse`) loss, epoch callbacks, and validation monitors.
  - Hardware adaptability: Executes seamlessly on CPU via oneDNN optimizations or on NVIDIA GPUs via CUDA/cuDNN.

### Feature Scaling & Preprocessing
- **`MinMaxScaler` (Scikit-Learn)**:
  - Normalizes features into the $[0, 1]$ interval. Essential for LSTM networks to avoid exploding/vanishing gradients and match the sigmoid/tanh activation boundaries.
- **`StandardScaler` (Scikit-Learn)**:
  - Standardizes features to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$). Crucial for distance-based algorithms like SVR.

### Interactive Web UI Framework
- **Gradio (`>= 4.0.0`)**:
  - Rapid machine learning deployment framework creating web interfaces directly within Python notebooks.
  - Components: `gr.Interface`, `gr.Blocks`, `gr.Textbox`, `gr.Number`, `gr.Slider`, `gr.Markdown`, and responsive form layouts.
  - Generates a local, low-latency web server (`http://127.0.0.1:7860`) with full offline execution (`share=False`).

### Model Persistence & Serialization
- **Joblib (`>= 1.3.0`)**:
  - Optimized binary serialization for NumPy-heavy Scikit-Learn pipelines and XGBoost estimators (`.pkl`).
- **Keras Native Serialization (`.keras`)**:
  - Single-file archive standard storing architecture, weight tensors, and optimizer configurations in a portable format.

### Data Visualization
- **Matplotlib (`>= 3.7.0`)**:
  - Generates static loss curves across training epochs, actual vs. predicted overlay plots, and residual distribution histograms.

### Development Environment & Extensions
- **VS Code (Visual Studio Code)**:
  - Primary recommended IDE with built-in Git tooling, integrated terminal, and interactive notebook viewer.
  - **VS Code Extensions**:
    - *Python Extension* (Microsoft): IntelliSense, linting, environment switching.
    - *Jupyter Extension* (Microsoft): Notebook execution, variable inspection, and cell debugging.
- **Jupyter (`>= 1.0.0`) & IPyKernel (`>= 6.20.0`)**:
  - Interactive computing environment allowing step-by-step block execution, cell-level inspection, and visual output rendering.

### Version Control & Large File Storage
- **Git**: Distributed source code version tracking.
- **Git LFS (Large File Storage)**: Git extension managing large binary assets (`.pkl`, `.keras`, `.csv`, `.xlsx`) as light text pointers in the repository history.

---

## 🤖 Machine Learning & Deep Learning Architectures

### 1. Ordinary Least Squares Linear Regression
- **Mathematical Principle**: Models the relationship between target $y$ and feature vector $\mathbf{x}$ as $y = \mathbf{w}^T \mathbf{x} + b$, minimizing residual sum of squares:
  $$\min_{\mathbf{w}} \sum_{i=1}^n \left( y_i - \mathbf{w}^T \mathbf{x}_i - b \right)^2$$
- **Role in Project**: Baseline benchmark to quantify how much non-linear algorithms improve forecasting accuracy over a linear hyperplane.

### 2. Random Forest Regressor
- **Mathematical Principle**: An ensemble bagging technique constructing $B$ de-correlated decision trees $\{T_1, T_2, \dots, T_B\}$. Predictions are averaged across all trees:
  $$\hat{y} = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$
- **Hyperparameters**: `n_estimators=100`, `max_depth=20`, `n_jobs=-1` (utilizes all CPU cores).
- **Strengths**: Robust against outliers, does not require strict feature scaling, and models non-linear feature interactions naturally.

### 3. Support Vector Regression (SVR)
- **Mathematical Principle**: Finds a function $f(\mathbf{x}) = \mathbf{w}^T \phi(\mathbf{x}) + b$ with at most $\epsilon$ deviation from actual targets $y_i$, penalizing errors larger than $\epsilon$ via slack variables $\xi_i, \xi_i^*$:
  $$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)$$
- **Kernel Function**: Radial Basis Function (RBF):
  $$K(\mathbf{x}, \mathbf{x}') = \exp\left( -\gamma \|\mathbf{x} - \mathbf{x}'\|^2 \right)$$
- **Preprocessing Requirement**: Strongly sensitive to feature magnitude; requires `StandardScaler` or `MinMaxScaler`.

### 4. XGBoost Regressor
- **Mathematical Principle**: Gradient boosted decision tree framework adding trees sequentially to minimize a regularized objective:
  $$\mathcal{L}^{(t)} = \sum_{i=1}^n l\left(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)\right) + \Omega(f_t)$$
  where $\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$.
- **Hyperparameters**: `n_estimators=300`, `learning_rate=0.05`, `max_depth=8`.
- **Strengths**: Fast histogram-based tree building, built-in L1/L2 shrinkage, and high predictive power on structured tabular lag data.

### 5. Long Short-Term Memory (LSTM) Networks
- **Mathematical Principle**: Special recurrent neural network designed to retain information across long sequences via gating mechanisms:
  - **Forget Gate**: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
  - **Input Gate**: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$, $\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$
  - **Cell State Update**: $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
  - **Output Gate**: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$, $h_t = o_t * \tanh(C_t)$
- **Architecture (Indian Dataset LSTM)**:
  - Input layer: 3D sequence tensor $(N, 24, 1)$ representing 24 historical hourly load observations.
  - Layer 1: `LSTM(64, return_sequences=True)` + `Dropout(0.2)`.
  - Layer 2: `LSTM(32, return_sequences=False)` + `Dropout(0.2)`.
  - Layer 3: `Dense(16, activation='relu')`.
  - Output layer: `Dense(1)` predicting the next hour's national load demand.
- **Optimizer**: Adam ($\text{lr} = 0.001$).
- **Loss**: Mean Squared Error ($\text{MSE}$).

---

## 📊 Model Performance Benchmark & Insights

### Foreign Dataset (Household Level)

| Model | Evaluation Framework | MAE (kW) | RMSE (kW) | $R^2$ Score | Key Observations |
|---|---|---|---|---|---|
| **XGBoost** | Lag Features (Gradio) | **0.0821** | **0.2161** | **0.9418** | **Best overall accuracy**; rapid convergence |
| **Random Forest** | Lag Features (Gradio) | 0.0829 | 0.2168 | 0.9414 | Competitive performance; resilient to lag noise |
| **Linear Regression** | Lag Features (Gradio) | 0.0850 | 0.2204 | 0.9395 | High baseline when provided autoregressive lag terms |
| **LSTM** | 60-step Window (Gradio) | 0.1145 | 0.2293 | 0.9345 | Effectively captures non-linear appliance cycles |
| **SVR** | Lag Features (Gradio) | 0.2190 | 0.3554 | 0.8427 | Accurate, but higher compute cost on large datasets |
| **Random Forest** | 30-step Window | 0.1668 | 0.3796 | 0.9354 | Effective on sliding window sequences |
| **XGBoost** | 30-step Window | 0.1765 | 0.3823 | 0.9345 | Fast inference across sequence matrices |
| **LSTM** | 30-step Window | 0.2577 | 0.4783 | 0.8714 | Smooth temporal generalization |
| **SVR** | 30-step Window | 0.2620 | 0.5781 | 0.8502 | Captures RBF boundaries on normalized series |

*(Note: In [`Linear Regression.ipynb`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/Linear%20Regression.ipynb), using concurrent circuit features like `Global_intensity` yields an $R^2$ of 0.9985 due to direct physical Ohm's law correlation $P \approx V \cdot I$).*

---

### Indian Dataset (National Grid Level)

| Model | Input Feature Strategy | MAE (MW) | RMSE (MW) | $R^2$ Score | Practical Assessment |
|---|---|---|---|---|---|
| **LSTM Deep Neural Network** | 24-step Sequential Load Window | **2,245.32** | **2,904.14** | **+0.9779** | **Superior Model.** Accurately models continuous grid dispatch and inertia |
| **Random Forest Regressor** | Calendar Timestamp Only | 16,030.64 | 19,612.99 | -0.0093 | Static calendar timestamps cannot capture rapid grid swings |
| **XGBoost Regressor** | Calendar Timestamp Only | 16,304.42 | 19,745.77 | -0.0230 | Captures rough mean but lacks autoregressive series context |
| **SVR (RBF Kernel)** | Standardized Calendar Features | 16,883.06 | 20,446.05 | -0.0968 | Smooth boundary mapping; limited by static calendar inputs |
| **Linear Regression** | Calendar Timestamp Only | 17,303.20 | 20,854.42 | -0.1411 | Linear fit fails across multi-seasonal national variations |

> [!TIP]
> **Key Scientific Takeaway**: On national electricity demand, sequential modeling (**LSTM**) outperforms static calendar regression by an immense margin ($R^2 = 0.9779$ vs negative $R^2$). This proves that national electric power dispatch is an autoregressive physical process governed by recent demand history rather than isolated calendar dates.

---

## 🚀 Step-by-Step Execution & Usage Guide

### Prerequisites
1. **Operating System**: Windows 10/11, macOS, or Linux.
2. **Python**: Python 3.10, 3.11, or 3.12 (check via `python --version`).
3. **IDE**: Visual Studio Code with the **Python** and **Jupyter** extensions installed.
4. **Git & Git LFS**: Ensure Git is installed (`git --version`).

---

### Step 1: Clone or Navigate to the Repository

Open your terminal or PowerShell and change into the project directory:

```powershell
# Windows PowerShell
cd "d:\Documents (D)\PBL_Model"
```

```bash
# macOS / Linux
cd /path/to/PBL_Model
```

---

### Step 2: Create and Activate a Virtual Environment

Isolate the project packages by creating a local `.venv` environment:

#### Windows (PowerShell)
```powershell
# Create the virtual environment
python -m venv .venv

# If script execution is restricted, temporarily enable it for this process:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate the virtual environment
.venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt)
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upon activation, your terminal prompt will display `(.venv)`.

---

### Step 3: Install Required Dependencies

Upgrade `pip` and install all necessary packages from [`requirements.txt`](file:///d:/Documents%20(D)/PBL_Model/requirements.txt):

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Verify Git LFS Assets (Optional but Recommended)

Because the project includes large datasets and pre-trained weights (`.pkl`, `.keras`, `.csv`), verify that Git LFS has pulled the complete binary files:

```bash
# Check if Git LFS is installed and initialized
git lfs install

# Pull all large files tracked by LFS
git lfs pull
```

---

### Step 5: Configure VS Code and Jupyter Kernel

1. Launch VS Code in the project root:
   ```bash
   code .
   ```
2. Open any `.ipynb` file (e.g., [`INDIAN_DATASET/LSTM_Gradio.ipynb`](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/LSTM_Gradio.ipynb)).
3. In the top-right corner of the notebook editor, click **Select Kernel**.
4. Select **Python Environments...** and choose `.venv (Python 3.x.x)`.
5. The kernel is now configured and ready to execute.

---

### Step 6: Executing Training Notebooks

If you wish to re-train any model from scratch:
1. Open the desired notebook:
   - For household power: open [`FOREIGN_DATASET/XGBoost.ipynb`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/XGBoost.ipynb) or [`FOREIGN_DATASET/LSTM.ipynb`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/LSTM.ipynb).
   - For national demand: open [`INDIAN_DATASET/LSTM_Gradio.ipynb`](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/LSTM_Gradio.ipynb).
2. Click **Run All** in the notebook toolbar (or run cells sequentially using `Shift + Enter`).
3. The notebook will automatically:
   - Resolve data paths resiliently using local checks.
   - Preprocess and engineer features.
   - Train the estimator and display training loss/metrics.
   - Generate evaluation graphs (Actual vs. Predicted curves).
   - Save the trained model to its designated output directory.

---

### Step 7: Launching and Using the Gradio Web Applications

Every `*_Gradio.ipynb` notebook includes an interactive user interface built with Gradio.

#### 1. Launching the App
1. Open any Gradio notebook:
   - [`INDIAN_DATASET/LSTM_Gradio.ipynb`](file:///d:/Documents%20(D)/PBL_Model/INDIAN_DATASET/LSTM_Gradio.ipynb) *(Recommended for national grid demand)*
   - [`FOREIGN_DATASET/XGBoost_Gradio.ipynb`](file:///d:/Documents%20(D)/PBL_Model/FOREIGN_DATASET/XGBoost_Gradio.ipynb) *(Recommended for household power)*
2. Run all cells up to the final launch cell:
   ```python
   demo.launch()
   ```
3. A local server URL will be printed in the output:
   ```
   * Running on local URL:  http://127.0.0.1:7860
   ```
4. Open `http://127.0.0.1:7860` in any web browser.

#### 2. Interacting with the Indian Dataset Gradio UI
- **Input Fields**:
  - `Date`: Enter a calendar date formatted as `YYYY-MM-DD` (e.g., `2024-04-01`).
  - `Time`: Enter the target hour formatted as `HH:MM` (e.g., `18:00`).
- **Processing**: The interface parses the timestamp, extracts the preceding 24-hour sequence tensor, feeds it through the LSTM network, and converts normalized predictions back to Megawatts.
- **Output**:
  ```text
  Predicted National Electricity Demand: 187,521.70 MW
  Dataset timestamp used: 2024-04-01 18:00:00
  ```

#### 3. Interacting with the Foreign Dataset Gradio UI
- **Tabular Models (XGBoost, RFR, SVR, LR)**:
  - Enter numeric values for active power lags: `Power_Lag_1` (kW), `Power_Lag_5` (kW), `Power_Lag_15` (kW), `Power_Lag_60` (kW), and `Power_Lag_1440` (kW).
  - Adjust the calendar sliders (`Year`, `Month`, `Day`, `Hour`, `Minute`, `Day_of_Week`).
  - Click **Submit** to receive the immediate forecasted consumption (e.g., `Predicted Global Active Power: 1.4520 kW`).
- **LSTM Sequence Model**:
  - Enter an integer sequence index ($\ge 60$).
  - Click **Submit** to view the actual vs. forecasted demand comparison.

> [!NOTE]
> To run a second Gradio app simultaneously on a different port, set the `server_port` parameter in the notebook cell:
> ```python
> demo.launch(server_port=7861)
> ```

---

### Step 8: Standalone Python Inference (Direct Model Loading)

You can load and use the pre-trained models directly in any custom Python script without opening Jupyter Notebooks.

#### Example A: Predicting National Grid Demand Using Pre-Trained LSTM (`.keras`)
```python
import numpy as np
import tensorflow as tf

# 1. Load the pre-trained Keras LSTM model
model_path = "INDIAN_DATASET/LSTM Model Data/lstm_model.keras"
model = tf.keras.models.load_model(model_path)

# 2. Prepare a sample input sequence of 24 hourly steps (scaled 0 to 1)
# Shape: (batch_size=1, time_steps=24, features=1)
sample_sequence = np.random.rand(1, 24, 1).astype(np.float32)

# 3. Predict the next hour's normalized load
predicted_scaled = model.predict(sample_sequence)

# 4. Invert scaling using the dataset min/max values
# (e.g., min_load = 80,000 MW, max_load = 220,000 MW)
min_load, max_load = 80000.0, 220000.0
predicted_mw = predicted_scaled[0][0] * (max_load - min_load) + min_load
print(f"Predicted National Electricity Demand: {predicted_mw:,.2f} MW")
```

#### Example B: Predicting Household Consumption Using Pre-Trained XGBoost (`.pkl`)
```python
import joblib
import pandas as pd

# 1. Load the pre-trained Scikit-Learn/XGBoost model
model_path = "FOREIGN_DATASET/XGBoost_Output/xgboost_model.pkl"
model = joblib.load(model_path)

# 2. Construct feature dataframe matching the training schema
features = pd.DataFrame([{
    'power_lag_1': 1.35,
    'power_lag_5': 1.28,
    'power_lag_15': 1.10,
    'power_lag_60': 0.85,
    'power_lag_1440': 1.42,
    'Year': 2010,
    'Month': 10,
    'Day': 25,
    'Hour': 19,
    'Minute': 30,
    'Day_of_Week': 0
}])

# 3. Compute predicted active power
predicted_kw = model.predict(features)[0]
print(f"Predicted Active Power: {predicted_kw:.4f} kW")
```

---

## 🔧 Troubleshooting & FAQs

| Symptom / Error | Root Cause | Verified Resolution |
|---|---|---|
| `ModuleNotFoundError: No module named 'openpyxl'` | The Excel parsing engine is not installed in the active environment. | Run `pip install openpyxl` in your active `.venv`. |
| `OSError: [Errno 48] Address already in use: 7860` | An existing notebook kernel is already serving a Gradio app on port 7860. | Restart the prior notebook kernel or launch with a custom port: `demo.launch(server_port=7861)`. |
| `InconsistentVersionWarning: Trying to unpickle estimator...` | Model pickled under a differing `scikit-learn` minor version. | Safe non-fatal warning. If you wish to eliminate it, re-run the training notebook to overwrite the `.pkl` file under your current Scikit-Learn version. |
| Model files show small byte sizes (~130 bytes) and fail to load | Git LFS placeholder pointers are checked out instead of the binary data. | Run `git lfs install` followed by `git lfs pull` in the repository root. |
| `FileNotFoundError` when executing cells | Jupyter kernel started from a nested directory rather than workspace root. | All project notebooks contain dynamic path discovery (`Path.cwd()` and parent checking) that automatically detects file locations regardless of working directory. |
| High memory usage or slow training on `household_power_consumption.csv` | Full dataset contains ~2.075M records. | In training notebooks, adjust sample slicing (e.g., `df.iloc[:100000]`) or increase training batch sizes (`batch_size=64` or `128`) for rapid prototyping. |

---

## 📄 License & Attribution

- **Project**: Problem-Based Learning (PBL) Machine Learning Model Project
- **Datasets**:
  - Individual Household Electric Power Consumption: Public domain / UCI Machine Learning Repository.
  - National Load Dispatch Center (India): Public national grid records and Open Meteorological datasets.
- **Code & Pipelines**: Developed for portable, reproducible machine learning research and interactive deployment.
