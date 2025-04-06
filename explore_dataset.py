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
def visualize_distribution_original(data):
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

import os
import matplotlib.pyplot as plt
import random
from datetime import datetime

# Global state (you can replace this with session/counter logic)
VIS_INDEX_PATH = "static/results/vis_index.txt"

def get_next_index():
    if not os.path.exists(VIS_INDEX_PATH):
        with open(VIS_INDEX_PATH, "w") as f:
            f.write("0")
        return 0

    with open(VIS_INDEX_PATH, "r") as f:
        content = f.read().strip()
        if not content.isdigit():
            index = 0
        else:
            index = int(content)

    index = (index + 1) % 20

    with open(VIS_INDEX_PATH, "w") as f:
        f.write(str(index))

    return index


def visualize_distribution(data):
    index = get_next_index()

    # Random subset of the data (stratified)
    sample_frac = random.uniform(0.3, 0.7)
    data_sampled = data.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=sample_frac, random_state=index))

    label_counts = data_sampled['label'].value_counts()

    colors = ['skyblue', 'salmon', 'limegreen', 'gold', 'orchid', 'turquoise', 'orange']
    hatches = ['', '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    rotated = [0, 30, 45, 90]
    font_sizes = [10, 12, 14, 16]

    random.seed(index)
    color = random.sample(colors, k=2)
    hatch = random.sample(hatches, k=2)
    rot = random.choice(rotated)
    font = random.choice(font_sizes)
    sort_order = random.choice([True, False])

    counts = label_counts.sort_values(ascending=sort_order)

    plt.figure(figsize=(6 + random.random()*2, 4 + random.random()*2))
    bars = plt.bar(counts.index, counts.values, color=color)
    for bar, pattern in zip(bars, hatch):
        bar.set_hatch(pattern)

    plt.title(f'Distribution Style {index+1}', fontsize=font+2)
    plt.xlabel('Message Type', fontsize=font)
    plt.ylabel('Count', fontsize=font)
    plt.xticks(rotation=rot, fontsize=font)
    plt.yticks(fontsize=font)

    results_folder = "static/results"
    os.makedirs(results_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(results_folder, f'distribution_{index}_{timestamp}.png')
    plt.savefig(image_path)
    plt.close()
    return image_path
