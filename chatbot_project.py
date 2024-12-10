import pandas as pd
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify

# Download necessary NLTK resources
nltk.download('punkt')

# File paths
DATA_FILE = 'customer_support_data.csv'
MODEL_FILE = 'chatbot_model.pkl'


# Preprocessing Functionality
def preprocess_data(data):
    """
    Preprocess the dataset by converting text to lowercase and removing punctuation.
    """
    data['utterance'] = data['utterance'].str.lower()
    data['utterance'] = data['utterance'].str.replace(r'[^\w\s]', '', regex=True)  # Remove punctuation
    return data


# Model Training Functionality
def train_model(X_train, y_train):
    """
    Train a Naive Bayes model using TF-IDF vectorization.
    """
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    return model


# Chatbot Class
class Chatbot:
    """
    A simple chatbot class for handling user interactions.
    """
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def get_response(self, user_input):
        """
        Predict the intent of the user input and return a response.
        """
        intent = self.model.predict([user_input])[0]
        # Placeholder responses for intents (you can add actual responses here)
        responses = {
            "cancel_order": "You can cancel your order by going to the order section.",
            "delivery": "Delivery time usually takes 3-5 business days.",
            "contact_customer_service": "You can contact customer service at support@example.com.",
            "refund": "Refunds take 7-10 business days after approval.",
            "order_status": "You can check your order status in the 'My Orders' section."
        }
        return responses.get(intent, "Sorry, I didn't understand that.")


# Flask App for Deployment
app = Flask(__name__)
chatbot = None  # Placeholder for the chatbot instance


@app.route('/chat', methods=['POST'])
def chat():
    """
    Flask route to handle chat requests. Accepts a JSON object with 'message'.
    """
    user_input = request.json.get('message')
    response = chatbot.get_response(user_input)
    return jsonify({'response': response})


if __name__ == "__main__":
    # Load Dataset
    try:
        data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        # Sample dataset if the file is missing
        data = pd.DataFrame({
            'utterance': [
                "How can I cancel my order?",
                "What is the delivery time?",
                "How do I contact customer service?",
                "Can I get a refund?",
                "Where is my order?"
            ],
            'intent': [
                "cancel_order",
                "delivery",
                "contact_customer_service",
                "refund",
                "order_status"
            ]
        })
        data.to_csv(DATA_FILE, index=False)

    # Preprocess Data
    data = preprocess_data(data)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(data['utterance'], data['intent'], test_size=0.2, random_state=42)

    # Train Model
    model = train_model(X_train, y_train)

    # Save the Model
    joblib.dump(model, MODEL_FILE)

    # Load the chatbot with the trained model
    chatbot = Chatbot(MODEL_FILE)

    # Run Flask App
    print("Starting the chatbot server...")
    app.run(debug=True)
