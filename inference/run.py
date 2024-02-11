import pandas as pd
import joblib
import logging
import os
import time
import nltk
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

RESULTS_DIR = 'results'
LOG_FILE = os.path.join(RESULTS_DIR, 'inference_log.txt')
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, 'predictions.csv')

# Create the "results" folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_model(filepath='models/best_model.pkl'):
    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(script_dir, filepath)

    logging.info(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    model = joblib.load(model_path)

    logging.info("Model loaded successfully.")

    return model

def load_data(file_name):
    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(script_dir, "data")
    file_path = os.path.join(data_dir, file_name)
    return pd.read_csv(file_path)

def preprocess_text(text, use_stemming=True):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Removing punctuation and convert to lowercase
    tokens = [word for word in tokens if word.lower() not in stop_words]

    if use_stemming:
        tokens = [porter.stem(word) for word in tokens]  # Applying stemming
    else:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Applying lemmatization

    return ' '.join(tokens)

def preprocess_data(data, use_stemming=True):
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    data['preprocessed_text_stemming'] = data['review'].apply(lambda x: preprocess_text(x, use_stemming=True))
    data['preprocessed_text_lemmatization'] = data['review'].apply(lambda x: preprocess_text(x, use_stemming=False))

    return data

def run_inference():
    global stop_words, porter, lemmatizer
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    start_time = time.time()

    logger.info("Loading model")
    best_model = load_model()
    logger.info("Model loaded")

    logger.info("Loading data")
    data_test = load_data("test.csv")
    logger.info("Data loaded")
    logger.info("Preprocessing data")
    data_test = preprocess_data(data_test, use_stemming=True)
    logger.info("Data preprocessed")

    test_predictions_stem = best_model.predict(data_test['preprocessed_text_stemming'])
    test_accuracy_stem = accuracy_score(data_test['sentiment'], test_predictions_stem)
    logger.info(f'Test Accuracy with Actual Sentiments (Stemming): {test_accuracy_stem}')
    logger.info("Classification Report (Stemming):")
    logger.info(classification_report(data_test['sentiment'], test_predictions_stem))

    test_predictions_lemma = best_model.predict(data_test['preprocessed_text_lemmatization'])
    test_accuracy_lemma = accuracy_score(data_test['sentiment'], test_predictions_lemma)
    logger.info(f'Test Accuracy with Actual Sentiments (Lemmatization): {test_accuracy_lemma}')
    logger.info("Classification Report (Lemmatization):")
    logger.info(classification_report(data_test['sentiment'], test_predictions_lemma))
    end_time = time.time()

    # Store predictions in a CSV file
    predictions_df = pd.DataFrame({
        'Review': data_test['review'],
        'Actual_Sentiment': data_test['sentiment'],
        'Predicted_Sentiment_Stemming': test_predictions_stem,
        'Predicted_Sentiment_Lemmatization': test_predictions_lemma
    })
    predictions_df.to_csv(PREDICTIONS_FILE, index=False, columns=['Actual_Sentiment','Predicted_Sentiment_Stemming','Predicted_Sentiment_Lemmatization'])

    # Log elapsed time
    elapsed_time = end_time - start_time
    logger.info(f"Inference completed in {elapsed_time:.2f} seconds.")

    # Save all inference logs to a text file
    with open(LOG_FILE, 'a') as log_file:
        log_file.write("All Inference Logs:\n")
        log_file.write("-" * 50 + "\n")
        log_file.write(open(LOG_FILE, 'r').read())

    # Save accuracy results to a text file
    with open(os.path.join(RESULTS_DIR, 'accuracy_results.txt'), 'w') as accuracy_file:
        accuracy_file.write(f'Test Accuracy with Actual Sentiments (Stemming): {test_accuracy_stem}\n')
        accuracy_file.write(f'Test Accuracy with Actual Sentiments (Lemmatization): {test_accuracy_lemma}\n')
        accuracy_file.write("-" * 50 + "\n")
        accuracy_file.write(open(LOG_FILE, 'r').read())

if __name__ == "__main__":
    run_inference()
