import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from matplotlib.ticker import FuncFormatter
import ipaddress
import socket
import struct



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


# Module for loading CSV data
def load_data(file_path):
    try:
        logger.info(f"Attempting to load data from {file_path}")
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"No data in file: {file_path}")
        return None
    except pd.errors.ParserError:
        logger.error(f"Error parsing file: {file_path}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None

def clean_data(data):
    logger.info("Starting data cleaning process...")
    
    # Log initial data info
    logger.info(f"Initial data shape: {data.shape}")
    
    # Handling missing values
    logger.info("Checking for missing values...")
    missing_values = data.isnull().sum()
    logger.info(f"Missing values in each column:\n{missing_values[missing_values > 0]}")
    
    # Fill missing values with mean (for numeric columns)
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        if data[col].isnull().any():
            mean_value = data[col].mean()
            data[col].fillna(mean_value, inplace=True)
            logger.info(f"Filled missing values in '{col}' with mean: {mean_value}")

    # Removing duplicates
    logger.info("Checking for duplicates...")
    initial_shape = data.shape
    data.drop_duplicates(inplace=True)
    logger.info(f"Removed duplicates: {initial_shape[0] - data.shape[0]} rows removed.")
    return data







def convert_ip_to_integer(ecommerce_df, ip_country_df):
    logger.info("Convert IP addresses to integer format for both datasets...")
    ecommerce_df['ip_address'] = ecommerce_df['ip_address'].astype(int)
    ip_country_df['lower_bound_ip_address'] = ip_country_df['lower_bound_ip_address'].astype(int)
    return ecommerce_df, ip_country_df

def merge_datasets(ecommerce_df, ip_country_df):
    logger.info("Merge eCommerce data with IP country data based on IP ranges...")
    merged_data = pd.merge_asof(
            ecommerce_df.sort_values('ip_address'), 
            ip_country_df, 
            left_on='ip_address', 
            right_on='lower_bound_ip_address',
            direction='backward'
    )
    
    
    # Filter merged data for valid ranges
    merged_data = merged_data[
        (merged_data['ip_address'] >= merged_data['lower_bound_ip_address']) &
        (merged_data['ip_address'] <= merged_data['upper_bound_ip_address'])
    ]
    
    return merged_data

def save_results(merged_data, output_filepath):
    """Save the merged data to a CSV file."""
    merged_data.to_csv(output_filepath, index=False)

















































def data_overview(data):
    logger.info("Checking categorical values...")
    
    for col in data.select_dtypes(include=['object', 'category']).columns:
        unique_values = data[col].unique()
        logger.info(f"Unique values in '{col}': {unique_values}")

        # Count the frequency of each unique value
        value_counts = data[col].value_counts()
        logger.info(f"Value counts for '{col}':\n{value_counts}")

def outliers(data):
    logger.info("ploting box plot for detecting outliers")
    # Boxplot to visualize outliers in transaction amounts
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=data['Amount'])
    plt.title('Boxplot of Transaction Amounts')
    plt.ylabel('Transaction Amount')
     # Function to format y-axis labels
    def format_y_func(value, tick_number):
        return f'{int(value):,}'  # Format as integer with commas

    # Function to format x-axis labels
    def format_x_func(value, tick_number):
        return f'{int(value):,}'  # Format as integer with commas

    # Apply the custom formatter to the y-axis
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_func))

    # Apply the custom formatter to the x-axis
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_x_func))
    plt.show()
    logger.info("ploting box plot for detecting outliers")



def distribution_of_amount(data):
    logger.info("plotting distribution of amount...")
   
    # Create the plot
    plt.figure(figsize=(10, 4))
    sns.histplot(data['Amount'], bins=30, kde=True)

    # Set titles and labels
    plt.title('Distribution of Transaction Amounts')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Frequency')

    # Function to format y-axis labels
    def format_y_func(value, tick_number):
        return f'{int(value):,}'  # Format as integer with commas

    # Function to format x-axis labels
    def format_x_func(value, tick_number):
        return f'{int(value):,}'  # Format as integer with commas

    # Apply the custom formatter to the y-axis
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_func))

    # Apply the custom formatter to the x-axis
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_x_func))

    # Show the plot
    plt.show()
    logger.info("plot of distribution of amount")



def amount_by_fraud_result(data):
    # Boxplot for Amount by Fraud Result
    logger.info("plotting amount by fraud result...")

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='FraudResult', y='Amount', data=data)
    plt.title('Transaction Amount by Fraud Status')
    plt.xlabel('Fraud Status (0: No, 1: Yes)')
    plt.ylabel('Transaction Amount')
    plt.show()

    # Count plot for Fraud Results
    plt.figure(figsize=(10, 6))
    sns.countplot(x='FraudResult', data=data)
    plt.title('Count of Fraud Results')
    plt.xlabel('Fraud Status (0: No, 1: Yes)')
    plt.ylabel('Count')
    plt.show()
    logger.info("plotting amount by fraud result.")

def transaction_overtime(data):
    logger.info("ploting transaction trend over time...")
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

    # Group by date and sum amounts
    daily_transactions = data.groupby(data['TransactionStartTime'].dt.date)['Amount'].sum().reset_index()

    # Plot daily transaction amounts
    plt.figure(figsize=(12, 6))
    plt.plot(daily_transactions['TransactionStartTime'], daily_transactions['Amount'])
    plt.title('Daily Transaction Amounts Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Transaction Amount')
    plt.xticks(rotation=45)
    # Function to format y-axis labels
    def format_y_func(value, tick_number):
        return f'{int(value):,}'  # Format as integer with commas
    # Apply the custom formatter to the y-axis
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_func))
    plt.show()
    logger.info("plot of transaction trend over time...")



def plot_of_product_catagory(data):
    logger.info("plotting product cataogries by fraud...")
        # Count plot for Product Categories by Fraud Result
    plt.figure(figsize=(12, 6))
    sns.countplot(x='ProductCategory', hue='FraudResult', data=data)
    plt.title('Product Categories by Fraud Status')
    plt.xlabel('ProductCategory')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Fraud Status', loc='upper right', labels=['No', 'Yes'])
    plt.show()

    # Count plot for Channels by Fraud Result
    plt.figure(figsize=(12, 6))
    sns.countplot(x='ChannelId', hue='FraudResult', data=data)
    plt.title('Channels by Fraud Status')
    plt.xlabel('ChannelId')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Fraud Status', loc='upper right', labels=['No', 'Yes'])
    plt.show()
    logger.info("plot of product cataogries by fraud...")

def correlation_matrix(data):
    logger.info("plotting the correlation matrix....")
     # Encode categorical columns if necessary
    data['ProductId'] = data['ProductId'].astype('category').cat.codes
    data['ProductCategory'] = data['ProductCategory'].astype('category').cat.codes
    data['ChannelId'] = data['ChannelId'].astype('category').cat.codes  # Encode ChannelId if it's categorical

    correlation_data = data[['ProductId', 'ProductCategory', 'ChannelId', 'Amount', 'Value']]

    # Calculate the correlation matrix
    correlation_matrix = correlation_data.corr()

    # Create a heatmap
    plt.figure(figsize=(12, 10))  # Increased size for better readability
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation Heatmap of Selected Features')
    plt.show()
    logger.info("plot of  the correlation matrix")
