from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Hello, this is your Flask app running on Vercel!"})

if __name__ == '__main__':
    app.run()
