from flask import Flask, render_template, request, send_file
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
from dotenv import load_dotenv  # Import to load environment variables
from explore_dataset import load_data, visualize_distribution, clean_text  # Import functions
from read_email import read_email_view 

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()
email = os.getenv('EMAIL')
email_pass = os.getenv('EMAIL_PASS')

# Load the dataset
data = load_data('emails.csv')  # Load data using your function
data['message'] = data['message'].apply(clean_text)

# Prepare data for model training
X = data['message']
y = data['label']

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    image_path = ""

    # Pagination settings
    page = int(request.args.get('page', 1))
    per_page = 10
    total_rows = len(data)
    total_pages = (total_rows + per_page - 1) // per_page
    
    start_row = (page - 1) * per_page
    end_row = start_row + per_page
    paginated_data = data[start_row:end_row]

    if request.method == 'POST':
        message = request.form['message']
        cleaned_message = clean_text(message)
        message_vectorized = vectorizer.transform([cleaned_message])
        prediction = model.predict(message_vectorized)[0]
        
        # Generate and save plot
        image_path = visualize_distribution(data)  # Call to visualize_distribution

    return render_template('index.html', prediction=prediction, data=paginated_data, image_path=image_path, page=page, total_pages=total_pages)

@app.route('/read_email', methods=['GET', 'POST'])
def read_email():
    return read_email_view()

@app.route('/download_spam')
def download_spam():
    spam_data = data[data['label'] == 'spam']
    csv_path = 'static/results/spam_messages.csv'
    spam_data.to_csv(csv_path, index=False)
    return render_template('download.html')

@app.route('/download_spam_csv')
def download_spam_csv():
    return send_file('static/results/spam_messages.csv', as_attachment=True)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

