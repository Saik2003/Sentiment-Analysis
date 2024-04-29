
import io
import pickle
import re
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from flask import (Flask, jsonify, redirect, render_template, request,send_file, url_for)
from flask_cors import CORS
from flask_login import (LoginManager, UserMixin, current_user, login_required,login_user, logout_user)
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
from werkzeug.security import check_password_hash, generate_password_hash

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///your_database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "Sai Kumar"

CORS(app)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

migrate = Migrate(app, db)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    data = db.relationship('UserData', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class UserData(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    input_text = db.Column(db.Text, nullable=False)
    result = db.Column(db.Text, nullable=False)

    def __init__(self, user_id, input_text, result):
        self.user_id = user_id
        self.input_text = input_text
        self.result = result


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data = request.get_json()
        username = data.get("username")
        email = data.get("email")
        password = data.get("password")

        if not username or not email or not password:
            return jsonify({"error": "Missing username, email, or password"}), 400

        if (
                User.query.filter_by(username=username).first()
                or User.query.filter_by(email=email).first()
        ):
            return jsonify({"error": "Username or email already exists"}), 400

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        return jsonify({"message": "User created successfully"}), 201
    else:
        return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"error": "Missing username or password"}), 400

        user = User.query.filter_by(username=username).first()

        if not user or not user.check_password(password):
            return jsonify({"error": "Invalid username or password"}), 401

        login_user(user)
        return jsonify({"message": "Logged in successfully"}), 200
    else:
        return render_template("login.html")


@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        if "file" in request.files:
            # Bulk prediction from CSV file
            file = request.files["file"]
            data = pd.read_csv(file)
            predictions = []

            for text_input in data["text_column_name"]:
                blob = TextBlob(text_input)
                sentiment_polarity = blob.sentiment.polarity

                # Adjust sentiment based on polarity and magnitude
                if sentiment_polarity > 0:
                    predicted_sentiment = "POSITIVE"
                else:
                    # If sentiment polarity is close to zero, consider it as neutral
                    predicted_sentiment = "NEGATIVE"

                predictions.append(predicted_sentiment)

            data["predicted_sentiment"] = predictions
            predictions_csv = data.to_csv(index=False)

            response = BytesIO()
            response.write(predictions_csv.encode())
            response.seek(0)

            return send_file(
                response,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

        elif "text" in request.json:
            # Single string prediction
            text_input = request.json["text"]
            blob = TextBlob(text_input)
            sentiment_polarity = blob.sentiment.polarity

            # Adjust sentiment based on polarity and magnitude
            if sentiment_polarity > 0:
                predicted_sentiment = "POSITIVE"
            elif sentiment_polarity < 0:
                predicted_sentiment = "NEGATIVE"
            else:
                # If sentiment polarity is close to zero, consider it as neutral
                predicted_sentiment = "NEUTRAL"

            user = current_user
            new_data = UserData(user_id=str(user), input_text=text_input, result=predicted_sentiment)
            db.session.add(new_data)
            db.session.commit()
            db.session.refresh(new_data)
            
            return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(port=5000, debug=True)