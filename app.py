# from flask import Flask, render_template, request, send_file, flash, redirect, url_for
# import pandas as pd
# import os
# from werkzeug.utils import secure_filename
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from dotenv import load_dotenv
# from explore_dataset import load_data, visualize_distribution, clean_text
# from read_email import read_email_view
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# app = Flask(__name__)
# app.secret_key = 'secret-key-for-flashing'

# load_dotenv()
# email = os.getenv('EMAIL')
# email_pass = os.getenv('EMAIL_PASS')

# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# def train_model(df):
#     df['message'] = df['message'].apply(clean_text)
#     X = df['message']
#     y = df['label']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     vec = CountVectorizer()
#     X_train_vec = vec.fit_transform(X_train)
#     X_test_vec = vec.transform(X_test)
#     clf = MultinomialNB()
#     clf.fit(X_train_vec, y_train)
#     acc = accuracy_score(y_test, clf.predict(X_test_vec))
#     return clf, vec, round(acc * 100, 2)

# data = load_data('emails.csv')
# model, vectorizer, accuracy_percent = train_model(data)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     global model, vectorizer, data, accuracy_percent
#     prediction = ""
#     image_path = ""

#     page = int(request.args.get('page', 1))
#     per_page = 10
#     total_rows = len(data)
#     total_pages = (total_rows + per_page - 1) // per_page
#     start_row = (page - 1) * per_page
#     end_row = start_row + per_page
#     paginated_data = data[start_row:end_row]

#     if request.method == 'POST':
#         form_type = request.form.get('form_type')

#         if form_type == 'predict':
#             message = request.form.get('message', '').strip()
#             if message:
#                 cleaned_message = clean_text(message)
#                 message_vectorized = vectorizer.transform([cleaned_message])
#                 prediction = model.predict(message_vectorized)[0]
#                 image_path = visualize_distribution(data)

#         elif form_type == 'upload':
#             uploaded_file = request.files.get('dataset')
#             if uploaded_file and uploaded_file.filename.endswith('.csv'):
#                 filename = secure_filename(uploaded_file.filename)
#                 filepath = os.path.join(UPLOAD_FOLDER, filename)
#                 uploaded_file.save(filepath)
#                 print(f"[INFO] Uploaded file saved: {filepath}")
#                 flash(f'File {filename} uploaded successfully.', 'info')

#                 try:
#                     new_df = pd.read_csv(filepath)
#                     if 'label' in new_df.columns and 'message' in new_df.columns:
#                         model, vectorizer, accuracy_percent = train_model(new_df)
#                         data = new_df
#                         flash(f'Model retrained successfully with uploaded dataset. Accuracy: {accuracy_percent:.2f}%', 'success')
#                         return redirect(url_for('index'))
#                     else:
#                         flash('CSV must contain "label" and "message" columns.', 'danger')
#                 except Exception as e:
#                     flash(f'Error reading CSV file: {str(e)}', 'danger')
#             else:
#                 flash('Please upload a valid CSV file.', 'danger')

#     return render_template('index.html',
#                            prediction=prediction,
#                            data=paginated_data,
#                            image_path=image_path,
#                            page=page,
#                            total_pages=total_pages,
#                            accuracy=accuracy_percent)

# @app.route('/read_email', methods=['GET', 'POST'])
# def read_email():
#     return read_email_view()

# @app.route('/download_spam')
# def download_spam():
#     spam_data = data[data['label'] == 'spam']
#     csv_path = 'static/results/spam_messages.csv'
#     os.makedirs(os.path.dirname(csv_path), exist_ok=True)
#     spam_data.to_csv(csv_path, index=False)
#     return render_template('download.html')

# @app.route('/download_spam_csv')
# def download_spam_csv():
#     return send_file('static/results/spam_messages.csv', as_attachment=True)

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port, debug=True)




from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
from explore_dataset import load_data, visualize_distribution, clean_text
from read_email import read_email_view
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'secret-key-for-flashing'

load_dotenv()
email = os.getenv('EMAIL')
email_pass = os.getenv('EMAIL_PASS')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def train_model(df):
    df['message'] = df['message'].apply(clean_text)
    X = df['message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vec = CountVectorizer()
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_vec))
    return clf, vec, round(acc * 100, 2)

data = load_data('emails.csv')
model, vectorizer, accuracy_percent = train_model(data)

@app.route('/', methods=['GET', 'POST'])
def index():
    global model, vectorizer, data, accuracy_percent
    prediction = ""
    probability = None
    image_path = ""

    search_query = request.args.get('search', '').strip().lower()
    filtered_data = data.copy()
    if search_query:
        filtered_data = filtered_data[filtered_data['message'].str.lower().str.contains(search_query, na=False)]

    page = int(request.args.get('page', 1))
    per_page = 10
    total_rows = len(filtered_data)
    total_pages = (total_rows + per_page - 1) // per_page
    start_row = (page - 1) * per_page
    end_row = start_row + per_page
    paginated_data = filtered_data[start_row:end_row]

    if request.method == 'POST':
        form_type = request.form.get('form_type')

        if form_type == 'predict':
            message = request.form.get('message', '').strip()
            if message:
                cleaned_message = clean_text(message)
                message_vectorized = vectorizer.transform([cleaned_message])
                prediction = model.predict(message_vectorized)[0]
                probability = model.predict_proba(message_vectorized)[0].max() * 100
                image_path = visualize_distribution(data)

        elif form_type == 'upload':
            uploaded_file = request.files.get('dataset')
            if uploaded_file and uploaded_file.filename.endswith('.csv'):
                filename = secure_filename(uploaded_file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                uploaded_file.save(filepath)
                print(f"[INFO] Uploaded file saved: {filepath}")
                flash(f'File {filename} uploaded successfully.', 'info')

                try:
                    new_df = pd.read_csv(filepath)
                    if 'label' in new_df.columns and 'message' in new_df.columns:
                        model, vectorizer, accuracy_percent = train_model(new_df)
                        data = new_df
                        flash(f'Model retrained successfully with uploaded dataset. Accuracy: {accuracy_percent:.2f}%', 'success')
                        return redirect(url_for('index'))
                    else:
                        flash('CSV must contain "label" and "message" columns.', 'danger')
                except Exception as e:
                    flash(f'Error reading CSV file: {str(e)}', 'danger')
            else:
                flash('Please upload a valid CSV file.', 'danger')

    return render_template('index.html',
                           prediction=prediction,
                           probability=round(probability, 2) if probability else None,
                           data=paginated_data,
                           image_path=image_path,
                           page=page,
                           total_pages=total_pages,
                           accuracy=accuracy_percent,
                           search_query=search_query)

@app.route('/read_email', methods=['GET', 'POST'])
def read_email():
    return read_email_view()

@app.route('/download_spam')
def download_spam():
    spam_data = data[data['label'] == 'spam']
    csv_path = 'static/results/spam_messages.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    spam_data.to_csv(csv_path, index=False)
    return render_template('download.html')

@app.route('/download_spam_csv')
def download_spam_csv():
    return send_file('static/results/spam_messages.csv', as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)