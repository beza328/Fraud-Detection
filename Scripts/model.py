import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the encoder and scaler
encoder = LabelEncoder()
scaler = StandardScaler()

# Load datasets
logger.info("Loading datasets...")
creditcard_data = pd.read_csv('creditcard.csv')
fraud_data = pd.read_csv('Fraud_Data.csv')
logger.info("Datasets loaded successfully.")

# Data Preparation
logger.info("Separating features and target for both datasets...")
X_creditcard = creditcard_data.drop(columns=['Class'])
y_creditcard = creditcard_data['Class']
X_fraud = fraud_data.drop(columns=['class'])
y_fraud = fraud_data['class']

# Train-Test Split
logger.info("Performing train-test split for both datasets...")
X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(
    X_creditcard, y_creditcard, test_size=0.3, random_state=42)
X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(
    X_fraud, y_fraud, test_size=0.3, random_state=42)
logger.info("Train-test split completed.")

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'MLP': MLPClassifier()
}

# Training and evaluation function
def train_and_evaluate(models, X_train, X_test, y_train, y_test, dataset_name):
    results = {}
    logger.info(f"Starting model training and evaluation for {dataset_name} dataset.")
    for name, model in models.items():
        logger.info(f"Training {name} model...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        results[name] = report
        logger.info(f"{name} performance:\n{classification_report(y_test, predictions)}")
    logger.info(f"Model training and evaluation completed for {dataset_name} dataset.")
    return results

# Train and evaluate models on both datasets
creditcard_results = train_and_evaluate(models, X_train_credit, X_test_credit, y_train_credit, y_test_credit, "Credit Card")
fraud_results = train_and_evaluate(models, X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud, "Fraud Data")

# MLOps: Versioning and Experiment Tracking
mlflow.set_experiment("Fraud Detection Experiment")

logger.info("Starting MLflow tracking...")
with mlflow.start_run():
    for name, model in models.items():
        logger.info(f"Logging {name} model to MLflow...")
        model.fit(X_train_credit, y_train_credit)
        predictions = model.predict(X_test_credit)
        
        # Log model, parameters, and metrics
        mlflow.sklearn.log_model(model, name)
        mlflow.log_params(model.get_params())
        
        # Log metrics
        report = classification_report(y_test_credit, predictions, output_dict=True)
        for metric, score in report["weighted avg"].items():
            mlflow.log_metric(f"{name}_{metric}", score)
        logger.info(f"{name} model logged successfully.")
logger.info("MLflow tracking completed.")
