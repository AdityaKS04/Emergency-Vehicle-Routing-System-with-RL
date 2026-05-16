# Emergency Vehicle Routing System with Reinforcement Learning

## Overview

This project is an AI-powered Emergency Vehicle (EV) Routing System designed to reduce ambulance response time in urban traffic.

The system combines:

- Machine Learning (XGBoost, Tuned XGBoost, Random Forest) to predict traffic clearance distance
- Reinforcement Learning (DQN) to select the best traffic signal for preemption
- Incident Detection to simulate road blockages
- A* Routing to dynamically compute the shortest feasible route
- Streamlit and Plotly for interactive simulation and visualization
- MLOps practices such as experiment tracking, Git tags, reproducibility, and Docker

---

## Project Structure

```text
Emergency-Vehicle-Routing-System-with-RL/
├── app.py
├── requirements.txt
├── README.md
├── Dockerfile
├── .dockerignore
├── Traffic_preprocessed_EV_with_queue_augmented.csv
│
├── ml/
├── models/
├── preprocessing/
├── simulation/
├── train/
├── utils/
└── rl/
    ├── environment.py
    ├── train_dqn.py
    ├── signal_agent.py
    └── results/
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AdityaKS04/Emergency-Vehicle-Routing-System-with-RL.git
cd Emergency-Vehicle-Routing-System-with-RL
```

### 2. Create and Activate a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

The project uses:

```text
Traffic_preprocessed_EV_with_queue_augmented.csv
```

Target column:

```text
Clearance_Distance_km_final
```

---

# Training Machine Learning Models

## Train Baseline XGBoost

```bash
python train/train_xgb.py
```

Generated files:
- `xgb_clearance_model.joblib`
- `xgb_feature_columns.joblib`

Move them into the `models/` folder if they are saved elsewhere.

---

## Train Random Forest

```bash
python train/rf.py
```

Generated files:
- `rf_clearance_model.joblib`
- `rf_feature_columns.joblib`

Move them into the `models/` folder if needed.

---

## Train Tuned XGBoost (Optuna)

```bash
python train/xgb_tuned.py
```

Generated files:
- `xgb_clearance_model_tuned.joblib`
- `xgb_tuning_results.joblib`

---

# Training Reinforcement Learning Policies

```bash
python rl/train_dqn.py
```

Generated files:
- `rl/policy_v1.zip`
- `rl/policy_v2_explored.zip`
- `rl/results/results_policy_v1.csv`
- `rl/results/results_policy_v2_explored.csv`

---

# Running the Streamlit Application

```bash
python -m streamlit run app.py
```

Open:

http://localhost:8501

---

# MLOps Features

## Versioning

Git tags:
- `exp-qlearning-1`
- `exp-qlearning-2`

## Experiment Tracking

CSV files in `rl/results/` store:
- Run ID
- Episodes
- Average Reward
- Average Waiting Time
- Learning Rate
- Epsilon values

## Reproducibility

This README provides exact commands to:
- Install dependencies
- Train ML models
- Train RL policies
- Run the application

---

# Docker Support

## Build Docker Image

```bash
docker build -t ev-routing-rl .
```

## Run Docker Container

```bash
docker run -p 8501:8501 ev-routing-rl
```

Open:

http://localhost:8501

---

# requirements.txt

```txt
streamlit
numpy
pandas
networkx
plotly
scikit-learn
xgboost
joblib
optuna
stable-baselines3
gymnasium
torch
```

---

# System Workflow

1. User enters queue length and speed.
2. Incident detection simulates road blockages.
3. ML predicts clearance distance.
4. Road weights are updated.
5. A* computes the best route.
6. RL selects the optimal traffic signal.
7. Plotly animates the emergency vehicle.

---

# Abstract

This project presents an AI-based Emergency Vehicle Routing System that helps ambulances reach patients faster in urban traffic. Machine learning predicts how far traffic should be cleared, A* routing computes the shortest path, and reinforcement learning selects the best traffic signal to prioritize. The system is deployed as an interactive Streamlit application and incorporates MLOps practices such as experiment tracking, version control, reproducibility, and Docker support.

---

# Problem Statement

Emergency vehicles are often delayed by traffic congestion, red signals, and road incidents. Traditional navigation systems do not dynamically predict clearance requirements or control signals. This project addresses this challenge by predicting traffic clearance distance, rerouting around blocked roads, and using reinforcement learning to choose the optimal signal to create a green corridor.

---

# Tools Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Optuna
- Stable-Baselines3
- Gymnasium
- PyTorch
- NetworkX
- Streamlit
- Plotly
- Git and GitHub
- Docker

---

# Conclusion

This project demonstrates how machine learning, reinforcement learning, and graph algorithms can be integrated to optimize emergency vehicle routing. It reduces response time by predicting traffic clearance distance, dynamically rerouting around incidents, and selecting the best traffic signal. MLOps features ensure the project is reproducible, versioned, and deployment-ready.

---

# Author

Aditya KS
