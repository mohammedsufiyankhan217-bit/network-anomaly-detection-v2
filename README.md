# Network Anomaly Detection V2

A machine learning-powered cybersecurity dashboard that analyzes network traffic and identifies potentially malicious activity using the CIC-IDS2017 dataset.

## Project Overview

Network traffic can contain malicious activities such as DDoS attacks, Port Scanning, and Web Attacks. This project uses machine learning to analyze network flow data, classify traffic, and help visualize potential security threats through an interactive dashboard.

## Technologies Used

- Python
- Streamlit – Interactive dashboard
- Pandas & NumPy – Data processing
- Scikit-learn – Machine learning
- Plotly – Interactive visualizations
- Joblib – Saving and loading trained models

## Dataset

CIC-IDS2017 – A publicly available network intrusion detection dataset containing benign traffic and multiple types of cyberattacks.

The project uses the dataset's network flow features to train and evaluate machine learning models.

## Dashboard Modules

1. **Executive Overview** – Displays important network traffic statistics and security insights.
2. **Attack Distribution** – Visualizes the distribution of benign traffic and different attack categories.
3. **Dataset Explorer** – Allows users to explore and filter network traffic records.
4. **ML Training & Evaluation** – Presents model evaluation metrics and classification performance.
5. **Live Anomaly Detection** – Provides an interface to analyze selected or uploaded network flow data using trained models.
6. **Feature Importance** – Highlights the network traffic features that influence model predictions.
7. **Security Reports & Export** – Generates and exports analysis results.

## Machine Learning

The project uses Random Forest classification models for:

- Binary classification: Benign or Malicious traffic.
- Multiclass classification: Identifying different traffic and attack categories.

The models are trained using network flow features extracted from the CIC-IDS2017 dataset.

## Model Performance

Reported evaluation results:

| Metric | Multiclass Model | Binary Model |
|---|---:|---:|
| Accuracy | 99.46% | 99.73% |
| Balanced Accuracy | 91.14% | 99.51% |

*Results are based on the project's current evaluation pipeline.*

## Project Structure

```text
Network-Anomaly-Detection-V2/
├── app.py
├── requirements.txt
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── explainability.py
│   ├── model_trainer.py
│   └── reporting.py
├── views/
│   ├── overview.py
│   ├── attack_distribution.py
│   ├── dataset_explorer.py
│   ├── ml_evaluation.py
│   ├── detection.py
│   ├── feature_importance.py
│   └── reports_view.py
├── models/
└── data/
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/mohammedsufiyankhan217-bit/network-anomaly-detection-v2.git
```

2. Navigate into the project folder:

```bash
cd network-anomaly-detection-v2
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the dashboard:

```bash
streamlit run app.py
```

## Project Goal

To understand how machine learning can support network security by analyzing traffic patterns, detecting suspicious activity, evaluating classification models, and presenting security insights through an interactive dashboard.

---

**Note:** This is a dataset-based machine learning demonstration. It does not currently perform continuous live packet capture from a real network.
