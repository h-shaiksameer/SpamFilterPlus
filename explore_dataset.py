import matplotlib
matplotlib.use('Agg')


import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from datetime import datetime

# Function to clean the text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = text.strip()
    return text

# Function to load the dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
    except FileNotFoundError:
        print("The file 'emails.csv' was not found.")
        exit()
    except pd.errors.ParserError as e:
        print("Error parsing the CSV file:", e)
        exit()
    data['message'] = data['message'].apply(clean_text)
    return data

# Function to visualize the distribution
# Function to visualize the distribution
def visualize_distribution(data):
    data['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Distribution of Spam vs. Ham Messages')
    plt.xlabel('Message Type')
    plt.ylabel('Count')
    
    # Save in static/results directory
    results_folder = "static/results"  # Change the path
    os.makedirs(results_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(results_folder, f'distribution_plot_{timestamp}.png')
    plt.savefig(image_path)
    plt.close()
    return image_path

