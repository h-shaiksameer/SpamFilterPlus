import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tkinter as tk
from tkinter import scrolledtext
from PIL import ImageTk, Image
import os
from datetime import datetime

# Function to clean the text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = text.strip()
    return text

# Load the dataset
try:
    data = pd.read_csv('emails.csv', sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
except FileNotFoundError:
    print("The file 'emails.csv' was not found.")
    exit()
except pd.errors.ParserError as e:
    print("Error parsing the CSV file:", e)
    exit()

# Clean the message column
data['message'] = data['message'].apply(clean_text)

# Visualize the distribution
data['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Spam vs. Ham Messages')
plt.xlabel('Message Type')
plt.ylabel('Count')
plt.xticks(rotation=0)

# Create results folder if it doesn't exist
results_folder = "results"
os.makedirs(results_folder, exist_ok=True)

# Save the plot with a unique name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_path = os.path.join(results_folder, f'distribution_plot_{timestamp}.png')
plt.savefig(image_path)
plt.close()  # Close the plot to prevent it from displaying immediately
# Show the plot for confirmation
plt.imshow(plt.imread(image_path))
plt.axis('off')
plt.show()

# Create a Tkinter UI
root = tk.Tk()
root.title("Spam Filtering System")
root.geometry("600x600")

# Create a scrolled text area
output_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=15)
output_area.pack(pady=20)

# Display the dataset information
output_area.insert(tk.END, "Basic Information:\n")
output_area.insert(tk.END, str(data.info()) + "\n\n")

# Display the distribution of spam vs. ham messages
output_area.insert(tk.END, "Distribution of Messages:\n")
output_area.insert(tk.END, str(data['label'].value_counts()) + "\n\n")

# Load and display the saved plot in the UI
img = Image.open(image_path)
img = img.resize((400, 300), Image.LANCZOS)  # Resize the image
img_tk = ImageTk.PhotoImage(img)

panel = tk.Label(root, image=img_tk)
panel.image = img_tk  # Keep a reference
panel.pack(pady=10)

# Prepare data for model training
X = data['message']
y = data['label']

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Display evaluation results
output_area.insert(tk.END, "Accuracy: {}\n".format(accuracy))
output_area.insert(tk.END, "Classification Report:\n")
output_area.insert(tk.END, str(classification_report(y_test, y_pred)) + "\n")
output_area.insert(tk.END, "Confusion Matrix:\n")
output_area.insert(tk.END, str(confusion_matrix(y_test, y_pred)) + "\n")

# Define a function to predict new messages
def predict_message(message):
    cleaned_message = clean_text(message)
    message_vectorized = vectorizer.transform([cleaned_message])
    prediction = model.predict(message_vectorized)
    return prediction[0]

# Add an input area for user messages
input_area_label = tk.Label(root, text="Enter your message:")
input_area_label.pack(pady=(10, 0))

input_area = tk.Text(root, height=4, width=50)
input_area.pack(pady=10)

# Function to predict input message
def predict_input_message():
    input_message = input_area.get("1.0", tk.END).strip()
    if input_message:
        predicted_label = predict_message(input_message)
        output_area.insert(tk.END, "Input Message: '{}' - Predicted Label: {}\n".format(input_message, predicted_label))
    else:
        output_area.insert(tk.END, "Please enter a message to predict.\n")

# Add a button to trigger the prediction
predict_button = tk.Button(root, text="Predict Message", command=predict_input_message)
predict_button.pack(pady=5)

# Test the prediction function with new messages
new_messages = [
    "Hey, are we still on for dinner tomorrow?",
    "You have been selected for a chance to win a $1,000 gift card!",
    "Let's catch up soon! I miss you.",
    "Click this link to claim your free prize!",
]

for msg in new_messages:
    predicted_label = predict_message(msg)
    output_area.insert(tk.END, "Message: '{}' - Predicted Label: {}\n".format(msg, predicted_label))

# Start the Tkinter main loop
root.mainloop()
