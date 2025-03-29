import joblib
import os

def load_model(filename):
    """Load the model from a file."""
    if os.path.exists(filename):
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    else:
        print(f"File {filename} does not exist.")
        return None
# Load the vectorizer (ensure the vectorizer was saved during training)
def load_vectorizer(vectorizer_filename):
    """Load the vectorizer from a file."""
    if os.path.exists(vectorizer_filename):
        vectorizer = joblib.load(vectorizer_filename)
        print(f"Vectorizer loaded from {vectorizer_filename}")
        return vectorizer
    else:
        print(f"File {vectorizer_filename} does not exist.")
        return None

def preprocess_input(input_data):
    # """Preprocess the input data to match the model's expected format."""
    # Load the vectorizer
    vectorizer_filename = "model/vectorizer.pkl"  # Adjust the path if needed
    vectorizer = load_vectorizer(vectorizer_filename)
    
    if vectorizer is None:
        print("Vectorizer not found. Cannot preprocess input.")
        return None

    # Transform the input data using the vectorizer
    processed_data = vectorizer.transform([input_data])
    return processed_data

def predict(model, input_data):
    """Make predictions using the loaded model."""
    if model is None:
        print("No model is loaded. Cannot make predictions.")
        return None
    try:
        # Preprocess the input data
        processed_data = preprocess_input(input_data)
        predictions = model.predict(processed_data)
        print(f"Predictions: {predictions}")
        return predictions
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

if __name__ == "__main__":
    model_filename = "model/spam_classifier.pkl"
    model = load_model(model_filename)

    # Example input data
    input_data = "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."
    
    # Make predictions
    predict(model, input_data)