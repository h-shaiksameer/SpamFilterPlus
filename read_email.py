import os
import re
import imaplib
import email
from email.header import decode_header
from flask import Flask, render_template, request

app = Flask(__name__)

# Function to clean the text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = text.strip()
    return text

# Function to check for spam indicators
def is_spam(content):
    spam_keywords = ['congratulations', 'you won', 'prize', 'claim', 'click here']
    if any(keyword in content for keyword in spam_keywords):
        return True
    if re.search(r'http[s]?://[^\s]+', content):  # Check for URLs
        return True
    return False

# Function to read emails and check for spam
def read_emails(start_index, end_index):
    email_user = os.getenv("EMAIL")  # Read email from environment variable
    email_pass = os.getenv("EMAIL_PASS")  # Read password from environment variable
    predictions = []

    try:
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        mail.login(email_user, email_pass)
        mail.select("inbox")

        # Search for all emails
        status, messages = mail.search(None, 'ALL')
        email_ids = messages[0].split()
        total_emails = len(email_ids)

        # Limit the number of emails to read
        email_ids = email_ids[max(0, total_emails - end_index):total_emails - start_index]

        for email_id in email_ids:
            res, msg = mail.fetch(email_id, "(RFC822)")
            msg = email.message_from_bytes(msg[0][1])

            # Decode email subject
            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding if encoding else 'utf-8')

            # Get email body
            email_body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        email_body = part.get_payload(decode=True).decode()
                        break
            else:
                email_body = msg.get_payload(decode=True).decode()

            # Clean and check email content for spam
            cleaned_body = clean_text(email_body)
            spam_flag = is_spam(cleaned_body)
            prediction_label = "Spam" if spam_flag else "Safe"
            predictions.append((subject, prediction_label))

        mail.logout()
    except Exception as e:
        print("Failed to read emails:", str(e))

    return predictions

@app.route('/read_email', methods=['GET', 'POST'])
def read_email_view():
    page = request.args.get('page', 1, type=int)
    per_page = 20
    start_index = (page - 1) * per_page
    end_index = start_index + per_page

    predictions = []
    if request.method == 'POST':
        predictions = read_emails(start_index, end_index)

    return render_template('read_email.html', predictions=predictions, page=page)

if __name__ == '__main__':
    app.run(debug=True)
