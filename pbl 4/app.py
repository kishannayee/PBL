import joblib
from flask import Flask, render_template, request, jsonify
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Preprocess the input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]  # Keep only alphanumeric words
    text = [word for word in text if word not in stopwords.words('english')]  # Remove stopwords
    text = [ps.stem(word) for word in text]  # Apply stemming
    return " ".join(text)

# Load model and vectorizer
try:
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))  # Assuming the vectorizer is in 'vectorizer.pkl'
    model = pickle.load(open("model.pkl", "rb"))  # Assuming the model is in 'model.pkl'
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print("Error loading model or vectorizer:", str(e))

# Serve the frontend HTML
@app.route('/')
def index():
    return render_template('index.html')

# API for spam prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debugging: Log the incoming data
        data = request.get_json()  # Get the JSON data sent from frontend
        print("Received data:", data)  # Print received data to the console
        
        message = data.get('message', '')  # Extract the 'message' field

        if not message.strip():  # If the message is empty
            return jsonify({'error': 'Message is empty'}), 400

        # Debugging: Print the message and transformed text
        print("Original message:", message)
        
        transformed_sms = transform_text(message)  # Preprocess the message
        print("Transformed message:", transformed_sms)  # Print transformed text for debugging

        # Vectorize the input message using the vectorizer
        vector_input = vectorizer.transform([transformed_sms])
        print("Vectorized input:", vector_input.shape)  # Print the shape of the vectorized input for debugging

        # Make prediction
        prediction = model.predict(vector_input)[0]
        print("Prediction:", prediction)  # Print the prediction for debugging

        # Return the prediction result
        result = 'Spam' if prediction == 1 else 'Not Spam'
        return jsonify({'prediction': result})

    except Exception as e:
        print("Error occurred during prediction:", str(e))  # Print error in terminal for debugging
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
