import pandas as pd
import numpy as np
import logging
import os
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler


encoder = LabelEncoder()
scaler = StandardScaler()

# Define columns for scaling and encoding
categorical_columns = ['source', 'browser', 'sex']  # Categorical features
numeric_columns = [
        'purchase_value', 
        'transaction_frequency', 
        'average_transaction_value', 
        'time_since_signup', 
        'time_between_transactions'
    ]  # Numerical features


# Step 1: Define the path to the logs directory
log_dir = os.path.join(os.getcwd(), 'logs')  # Use current working directory

# Create the logs directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define file paths
log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

# Step 2: Create handlers
info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

# Step 3: Create a stream handler to output logs to the notebook
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)  # Output all logs to the notebook

# Step 4: Create a formatter and set it for the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Step 5: Create a logger and set its level
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Capture all logs (DEBUG and above)
logger.addHandler(info_handler)
logger.addHandler(error_handler)
logger.addHandler(stream_handler)  # Add stream handler for notebook output



def create_time_features(df):
    """Creates time-based features."""
    logger.info("Creating time-based features...")
    df['transaction_hour'] = df['purchase_time'].dt.hour
    df['transaction_day'] = df['purchase_time'].dt.dayofweek
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    logger.info("Time-based features created successfully.")
    return df



def create_user_behavior_features(df):
    """Creates features based on user transaction frequency and average purchase value."""
    logger.info("Creating user behavior features...")
    df['transaction_frequency'] = df.groupby('user_id')['purchase_time'].transform('count')
    df['average_transaction_value'] = df.groupby('user_id')['purchase_value'].transform('mean')
    logger.info("User behavior features created successfully.")
    return df



def create_ip_and_device_features(df):
    """Calculates the frequency of transactions by IP and device, often a fraud indicator."""
    logger.info("Creating IP and device features...")
    df['ip_transaction_count'] = df.groupby('ip_address')['purchase_time'].transform('count')
    df['device_id_frequency'] = df.groupby('device_id')['user_id'].transform('count')
    logger.info("IP and device features created successfully.")
    return df




def create_velocity_features(df):
    """Calculates the time difference between consecutive transactions for each user."""
    logger.info("Creating transaction velocity features...")
    df = df.sort_values(['user_id', 'purchase_time'])
    df['time_between_transactions'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds()
    logger.info("Transaction velocity features created successfully.")
    return df



def create_geolocation_features(df):
    """Creates a feature indicating country mismatch if the necessary columns are available."""
    logger.info("Creating geolocation features...")
  
    df['country_mismatch'] = (df['country'] != df['country']).astype(int)
    logger.info("Geolocation mismatch feature created successfully.")
    return df





def encode_and_scale_features(df, categorical_columns, numeric_columns):
    """Encodes categorical features and scales numerical features."""
    logger.info("Encoding categorical features and scaling numerical features...")
    
    # Encode categorical columns
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])
    
    # Scale numerical columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    logger.info("Encoding and scaling completed successfully.")
    return df


# Columns for scaling and encoding
categorical_columns = ['source', 'browser', 'sex']
numeric_columns = ['purchase_value', 'transaction_frequency', 'average_transaction_value', 'time_since_signup', 'time_between_transactions']

logger.info("Starting feature engineering pipeline...")


