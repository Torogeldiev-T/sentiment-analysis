import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
import logging
import subprocess
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(file_path):
    return pd.read_csv(file_path)

def explore_data(data):
    logger.info(data.head())
    logger.info(data.info())
    logger.info(data.describe())

def visualize_class_distribution(data):
    sns.countplot(x='sentiment', data=data)
    plt.title('Class Distribution')
    plt.show()

def visualize_review_length_distribution(data):
    data['review_length'] = data['review'].apply(len)
    sns.histplot(data=data, x='review_length', hue='sentiment', kde=True)
    plt.title('Review Length Distribution by Sentiment')
    plt.show()

def create_wordcloud(reviews, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(reviews))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

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

def split_data(data):
    X_train_stem, X_test_stem, y_train, y_test = train_test_split(data['preprocessed_text_stemming'], data['sentiment'], test_size=0.2, random_state=42)
    X_train_lemma, X_test_lemma, y_train, y_test = train_test_split(data['preprocessed_text_lemmatization'], data['sentiment'], test_size=0.2, random_state=42)
    return X_train_stem, X_test_stem, y_train, y_test, X_train_lemma, X_test_lemma

def train_and_evaluate_models(X_train, X_test, y_train, y_test, model_name, model, vectorizer):
    pipeline = make_pipeline(vectorizer, model) if vectorizer else make_pipeline(model)

    param_grid = {}
    if 'Logistic Regression' in model_name:
        param_grid = {'logisticregression__C': [0.1, 1, 10]}

    if 'Random Forest' in model_name:
        param_grid = {'randomforestclassifier__n_estimators': [10, 20, 30]}

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f'{model_name} Accuracy: {accuracy}')

    return grid_search.best_estimator_, accuracy

def run_data_generation_script():
    logger.info("Running data generation script...")
    try:
        subprocess.run(["python3", "data_process/data_generation.py"])
    except Exception as e:
        logger.error(f"Error running data generation script: {str(e)}")

def load_data(file_name):
    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(script_dir, "data")
    file_path = os.path.join(data_dir, file_name)

    return pd.read_csv(file_path)

def main():
    logger.info('Downloading stopwords')
    global stop_words, porter, lemmatizer
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    logger.info('Downloading train data')
    run_data_generation_script()

    data_train = load_data("train.csv")
    data_test = load_data("test.csv")

    explore_data(data_train)
    visualize_class_distribution(data_train)
    visualize_review_length_distribution(data_train)

    positive_reviews = data_train[data_train['sentiment'] == 'positive']['review']
    negative_reviews = data_train[data_train['sentiment'] == 'negative']['review']

    create_wordcloud(positive_reviews, 'Word Cloud for Positive Reviews')
    create_wordcloud(negative_reviews, 'Word Cloud for Negative Reviews')

    logger.info('Preprocessing train data')
    data_train = preprocess_data(data_train, use_stemming=True)
    
    X_train_stem, X_test_stem, y_train, y_test, X_train_lemma, X_test_lemma = split_data(data_train)

    models = {
        'Logistic Regression (CountVectorizer, Stemming)': LogisticRegression(max_iter=10000),
        'Logistic Regression (TfidfVectorizer, Stemming)': LogisticRegression(max_iter=10000),
        'Logistic Regression (CountVectorizer, Lemmatization)': LogisticRegression(max_iter=10000),
        'Logistic Regression (TfidfVectorizer, Lemmatization)': LogisticRegression(max_iter=10000),
        'Multinomial Naive Bayes (CountVectorizer, Stemming)': MultinomialNB(),
        'Multinomial Naive Bayes (TfidfVectorizer, Stemming)': MultinomialNB(),
        'Multinomial Naive Bayes (CountVectorizer, Lemmatization)': MultinomialNB(),
        'Multinomial Naive Bayes (TfidfVectorizer, Lemmatization)': MultinomialNB(),
        'Random Forest (CountVectorizer, Stemming)': RandomForestClassifier(n_estimators=20),
        'Random Forest (TfidfVectorizer, Stemming)': RandomForestClassifier(n_estimators=20),
        'Random Forest (CountVectorizer, Lemmatization)': RandomForestClassifier(n_estimators=20),
        'Random Forest (TfidfVectorizer, Lemmatization)': RandomForestClassifier(n_estimators=20)
    }

    best_model = None
    best_accuracy = 0

    for model_name, model in models.items():
        if 'CountVectorizer' in model_name:
            vectorizer = CountVectorizer()
        elif 'TfidfVectorizer' in model_name:
            vectorizer = TfidfVectorizer()
        else:
            vectorizer = None

        if 'Stemming' in model_name:
            X_train = X_train_stem
            X_test = X_test_stem
        elif 'Lemmatization' in model_name:
            X_train = X_train_lemma
            X_test = X_test_lemma

        trained_model, accuracy = train_and_evaluate_models(X_train, X_test, y_train, y_test, model_name, model, vectorizer)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = trained_model

    data_test = preprocess_data(data_test, use_stemming=True)

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

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)  # Create the 'models' folder if it doesn't exist
    best_model_path = os.path.join(model_dir, "best_model.pkl")

    joblib.dump(best_model, best_model_path)
    logger.info(f"Best model has been saved to: {best_model_path}")

if __name__ == "__main__":
    main()
