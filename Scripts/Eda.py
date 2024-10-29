import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from matplotlib.ticker import FuncFormatter
import geopandas as gpd



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


def calculate_class_distribution(df):
    logger.info("Calculate the distribution of the 'class' column.")
    class_distribution = df['class'].value_counts()
    fraud_percentage = (class_distribution / class_distribution.sum()) * 100
    return class_distribution, fraud_percentage

def plot_class_distribution(df, class_distribution):
    logger.info("Plotting class distribution...")


    # Bar plot
    plt.figure(figsize=(8, 6))
    sns.countplot(x='class', data=df, palette='Set2')
    plt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')
    plt.xlabel('Class (0 = Non-Fraud, 1 = Fraud)')
    plt.ylabel('Number of Transactions')
    plt.xticks(ticks=[0, 1], labels=['Non-Fraud', 'Fraud'], rotation=0)
    plt.show()

    # Pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(class_distribution, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90, colors=['#66c2a5','#fc8d62'])
    plt.title('Fraudulent vs Non-Fraudulent Transactions Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular
    plt.show()






def calculate_source_distribution(df):
  
    logger.info("Calculating source distribution...")  # Log the start of the calculation
    
    # Count occurrences of each source
    source_distribution = df['source'].value_counts()
    
    # Calculate percentages
    source_percentage = (source_distribution / source_distribution.sum()) * 100  
    
    # Log the distribution details
    logger.info("Source distribution calculated: %s", source_distribution.to_dict())
    
    return source_distribution, source_percentage  # Return the distribution and percentages

def plot_source_distribution(source_distribution):
  
    logger.info("Plotting source distribution...")  # Log the start of the plotting process

    # Create a bar plot
    plt.figure(figsize=(10, 6))  # Create a new figure with a specified size
    sns.barplot(y=source_distribution.index, x=source_distribution.values, palette='Set2')  # Create a count plot for 'source'
    plt.title('Distribution of Transaction Sources')  # Set the title of the plot
    plt.xlabel('Number of Transactions')  # Set the x-axis label
    plt.ylabel('Source')  # Set the y-axis label
    plt.show()  # Display the plot

    # Create a pie chart for better visualization of proportions
    plt.figure(figsize=(8, 6))  # Create a new figure for the pie chart
    plt.pie(source_distribution, labels=source_distribution.index, autopct='%1.1f%%', startangle=90)  # Create a pie chart
    plt.title('Transaction Sources Distribution')  # Set the title of the pie chart
    plt.axis('equal')  # Set the aspect ratio of the pie chart to be equal
    plt.show()  # Display the pie chart

    logger.info("Source distribution plotted successfully.")  # Log the successful plotting


def calculate_sex_distribution(df):
    
    logger.info("Calculating sex distribution...")  # Log the start of the calculation
    
    # Count occurrences of each sex
    sex_distribution = df['sex'].value_counts()
    
    # Calculate percentages
    sex_percentage = (sex_distribution / sex_distribution.sum()) * 100  
    
    # Log the distribution details
    logger.info("Sex distribution calculated: %s", sex_distribution.to_dict())
    
    return sex_distribution, sex_percentage  # Return the distribution and percentages

def plot_sex_distribution(sex_distribution):
   
    logger.info("Plotting sex distribution...")  # Log the start of the plotting process

    # Create a bar plot
    plt.figure(figsize=(10, 6))  # Create a new figure with a specified size
    sns.barplot(y=sex_distribution.index, x=sex_distribution.values, palette='Set2')  # Create a count plot for 'sex'
    plt.title('Distribution of Transaction Genders')  # Set the title of the plot
    plt.xlabel('Number of Transactions')  # Set the x-axis label
    plt.ylabel('Gender')  # Set the y-axis label
    plt.show()  # Display the plot

    # Create a pie chart for better visualization of proportions
    plt.figure(figsize=(8, 6))  # Create a new figure for the pie chart
    plt.pie(sex_distribution, labels=sex_distribution.index, autopct='%1.1f%%', startangle=90)  # Create a pie chart
    plt.title('Transaction Genders Distribution')  # Set the title of the pie chart
    plt.axis('equal')  # Set the aspect ratio of the pie chart to be equal
    plt.show()  # Display the pie chart

    logger.info("Sex distribution plotted successfully.")  # Log the successful plotting



def calculate_age_statistics(df):
   
    logger.info("Calculating age statistics...")  # Log the start of the calculation
    
    # Calculate statistics
    age_mean = df['age'].mean()  # Calculate mean age
    age_median = df['age'].median()  # Calculate median age
    age_std = df['age'].std()  # Calculate standard deviation of age
    
    # Log the statistics
    logger.info("Age statistics calculated: Mean=%.2f, Median=%.2f, Std Dev=%.2f", age_mean, age_median, age_std)
    
    return pd.Series({'mean': age_mean, 'median': age_median, 'std_dev': age_std})  # Return the statistics

def plot_age_distribution(df):
   
    logger.info("Plotting age distribution...")  # Log the start of the plotting process

    # Create a histogram
    plt.figure(figsize=(10, 6))  # Create a new figure with a specified size
    sns.histplot(df['age'], bins=30, kde=True, color='blue', stat='density')  # Create a histogram with density estimation
    plt.title('Age Distribution of Transactions')  # Set the title of the histogram
    plt.xlabel('Age')  # Set the x-axis label
    plt.ylabel('Density')  # Set the y-axis label
    plt.axvline(df['age'].mean(), color='red', linestyle='dashed', linewidth=1)  # Add a line for mean age
    plt.axvline(df['age'].median(), color='green', linestyle='dashed', linewidth=1)  # Add a line for median age
    plt.legend({'Mean': df['age'].mean(), 'Median': df['age'].median()})  # Add legend
    plt.show()  # Display the histogram

    # Create a box plot
    plt.figure(figsize=(10, 6))  # Create a new figure for the box plot
    sns.boxplot(x=df['age'], color='cyan')  # Create a box plot for age
    plt.title('Box Plot of Age Distribution')  # Set the title of the box plot
    plt.xlabel('Age')  # Set the x-axis label
    plt.show()  # Display the box plot

    logger.info("Age distribution plotted successfully.")  # Log the successful plotting





# Bivarieate analysis

def analyze_fraud_by_source(df):
    
    logger.info("Analyzing fraudulent transactions by source...") 

    # Calculate the fraud rate for each source
    fraud_counts = df.groupby('source')['class'].value_counts().unstack()  # Count of fraud and non-fraud transactions
    #fraud_counts = fraud_counts.fillna(0)  # Fill NaN values with 0
    fraud_counts['fraud_rate'] = fraud_counts[1] / (fraud_counts[0] + fraud_counts[1])  # Calculate fraud rate
    fraud_counts = fraud_counts.reset_index()  # Reset index for easier plotting

    # Log the calculated fraud rates
    logger.info("Fraud rates by source:\n%s", fraud_counts[['source', 'fraud_rate']])  # Log fraud rates

    # Create a bar plot for fraud rates by source
    plt.figure(figsize=(12, 6))  # Create a new figure with a specified size
    sns.barplot(data=fraud_counts, x='source', y='fraud_rate', palette='viridis')  # Create a bar plot
    plt.title('Fraud Rate by Source')  # Set the title of the plot
    plt.xlabel('Source')  # Set the x-axis label
    plt.ylabel('Fraud Rate')  # Set the y-axis label
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.axhline(0.5, color='red', linestyle='--', linewidth=1)  # Add a horizontal line at fraud rate of 0.5
    plt.show()  # Display the plot

    logger.info("Bivariate analysis by source completed successfully.")  # Log completion of analysis

def analyze_fraud_by_sex(df):
 
    logger.info("Analyzing fraudulent transactions by sex...")  # Log the start of analysis

    # Calculate the fraud rate for each sex
    fraud_counts = df.groupby('sex')['class'].value_counts().unstack()  # Count of fraud and non-fraud transactions
    #fraud_counts = fraud_counts.fillna(0)  # Fill NaN values with 0
    fraud_counts['fraud_rate'] = fraud_counts[1] / (fraud_counts[0] + fraud_counts[1])  # Calculate fraud rate
    fraud_counts = fraud_counts.reset_index()  # Reset index for easier plotting

    # Log the calculated fraud rates
    logger.info("Fraud rates by sex:\n%s", fraud_counts[['sex', 'fraud_rate']])  # Log fraud rates

    # Create a bar plot for fraud rates by sex
    plt.figure(figsize=(8, 6))  # Create a new figure with a specified size
    sns.barplot(data=fraud_counts, x='sex', y='fraud_rate', palette='pastel')  # Create a bar plot
    plt.title('Fraud Rate by Sex')  # Set the title of the plot
    plt.xlabel('Sex')  # Set the x-axis label
    plt.ylabel('Fraud Rate')  # Set the y-axis label
    plt.axhline(0.5, color='red', linestyle='--', linewidth=1)  # Add a horizontal line at fraud rate of 0.5
    plt.show()  # Display the plot

    logger.info("Bivariate analysis by sex completed successfully.")  



# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_age_bins(df):
  
    # Define the bins and labels for age ranges
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Adjust upper limit as necessary
    labels = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    
    # Create a new column 'age_range' based on bins
    df['age_range'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)  # Use right=False for inclusive lower bound
    return df

def analyze_fraud_by_age_range(df):
   
    logger.info("Analyzing fraudulent transactions by age range...")  # Log the start of analysis

    # Create age bins
    df = create_age_bins(df)

    # Calculate the fraud rate for each age range
    fraud_counts = df.groupby('age_range')['class'].value_counts().unstack()  # Count of fraud and non-fraud transactions
    fraud_counts = fraud_counts.fillna(0)  # Fill NaN values with 0
    fraud_counts['fraud_rate'] = fraud_counts[1] / (fraud_counts[0] + fraud_counts[1])  # Calculate fraud rate
    fraud_counts = fraud_counts.reset_index()  # Reset index for easier plotting

    # Log the calculated fraud rates
    logger.info("Fraud rates by age range:\n%s", fraud_counts[['age_range', 'fraud_rate']])  # Log fraud rates

    # Create a bar plot for fraud rates by age range
    plt.figure(figsize=(12, 6))  # Create a new figure with a specified size
    sns.barplot(data=fraud_counts, x='age_range', y='fraud_rate', palette='pastel')  # Create a bar plot
    plt.title('Fraud Rate by Age Range')  # Set the title of the plot
    plt.xlabel('Age Range')  # Set the x-axis label
    plt.ylabel('Fraud Rate')  # Set the y-axis label
    plt.axhline(0.5, color='red', linestyle='--', linewidth=1)  # Add a horizontal line at fraud rate of 0.5
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()  # Display the plot

    logger.info("Bivariate analysis by age range completed successfully.")  # Log completion of analysis



# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_fraud_by_country(ecommerce_df):
 
    logger.info("Calculating fraud by country...")
    country_fraud_stats = (
        ecommerce_df.groupby('country')['class']
        .agg(total_transactions='count', total_fraud='sum')
        .assign(fraud_rate=lambda x: x['total_fraud'] / x['total_transactions'])
        .reset_index()
    )
    logger.info("Fraud calculation by country completed.")
    return country_fraud_stats

def top_fraudulent_countries(country_fraud_stats, top_n=10):
   
    logger.info(f"Extracting top {top_n} countries with highest fraud rates...")
    top_countries = country_fraud_stats.sort_values(by='fraud_rate', ascending=False).head(top_n)
    logger.info("Top fraudulent countries extraction completed.")
    return top_countries

def plot_fraud_map(country_fraud_stats, world_map_path='path_to_world_shapefile'):

    logger.info("Plotting fraud distribution on world map...")
    # Load the world shapefile data
    world = gpd.read_file(world_map_path)
    # Merge world map with fraud data
    fraud_map = world.merge(country_fraud_stats, how='left', left_on='NAME_LONG', right_on='country')

    # Plot the world map
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fraud_map.plot(column='fraud_rate', cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
    ax.set_title("Fraud Rate by Country", fontsize=16)
    ax.set_axis_off()
    plt.show()
    logger.info("Fraud map plotted successfully.")



