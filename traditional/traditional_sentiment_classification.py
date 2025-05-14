import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import contractions

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

# Load the dataset
data = pd.read_csv('../cafe_sentiment_dataset.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(f"Shape: {data.shape}")
print("\nColumns in the dataset:")
print(data.columns.tolist())
print("\nSample data:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Get the column names from the actual dataset
columns = data.columns.tolist()
print(f"\nDetected columns: {columns}")

# Identify text and sentiment columns
text_column = columns[0] 
sentiment_column = columns[1]

print(f"\nUsing '{text_column}' as the text column")
print(f"Using '{sentiment_column}' as the sentiment column")

# Check the distribution of sentiment classes
print("\nSentiment Distribution:")
sentiment_counts = data[sentiment_column].value_counts()
print(sentiment_counts)

# Text preprocessing function
def preprocess_text(text):
    """
    Function to clean and preprocess text data
    """
    if pd.isna(text):
        return ""
    
    # Convert text to lowercase
    text = text.lower()
    
    # Expand contractions (e.g., "can't" to "cannot")
    text = contractions.fix(text)
    
    # Handle numbers (replace with word 'number')
    text = re.sub(r'\d+', 'number', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into text
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

# Apply preprocessing to the text
print("\nApplying text preprocessing...")
data['cleaned_text'] = data[text_column].apply(preprocess_text)

# Display a few examples of cleaned text
print("\nExample of original vs cleaned text:")
for i in range(min(5, len(data))):
    print(f"Original: {data[text_column].iloc[i]}")
    print(f"Cleaned: {data['cleaned_text'].iloc[i]}")
    print("-" * 50)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned_text'], 
    data[sentiment_column], 
    test_size=0.2, 
    random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Build the TF-IDF + Naive Bayes pipeline
print("\nBuilding and training the TF-IDF + Naive Bayes model...")
tfidf_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),  # Using TF-IDF 
    ('classifier', MultinomialNB())
])

# Train the model
tfidf_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = tfidf_pipeline.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f} or {accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

precision = precision_score(y_test, y_pred, pos_label='positive')
recall = recall_score(y_test, y_pred, pos_label='positive')
f1 = f1_score(y_test, y_pred, pos_label='positive')

print("\nAdditional Metrics:")
print(f"Precision: {precision:.2f} or {precision*100:.2f}%")
print(f"Recall: {recall:.2f} or {recall*100:.2f}%")
print(f"F1-Score: {f1:.2f} or {f1*100:.2f}%")

# Feature importance analysis
vectorizer = tfidf_pipeline.named_steps['vectorizer']
classifier = tfidf_pipeline.named_steps['classifier']

# Get feature importance for each class
def get_most_informative_features(vectorizer, classifier, n=20):
    feature_names = vectorizer.get_feature_names_out()
    class_labels = classifier.classes_
    
    for i, class_label in enumerate(class_labels):
        top_indices = classifier.feature_log_prob_[i].argsort()[-n:][::-1]
        top_features = [(feature_names[j], classifier.feature_log_prob_[i][j]) for j in top_indices]
        
        print(f"\nTop {n} features for class '{class_label}':")
        for feature, importance in top_features:
            print(f"{feature}: {importance:.4f}")

# Print the most informative features
get_most_informative_features(vectorizer, classifier)

# Classify new text
def classify_sentiment(text, pipeline=tfidf_pipeline):

    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Make prediction
    sentiment = pipeline.predict([processed_text])[0]
    
    # Get probability scores
    probabilities = pipeline.predict_proba([processed_text])[0]
    prob_dict = {str(classifier.classes_[i]): float(prob) for i, prob in enumerate(probabilities)}
    
    return {
        'sentiment': sentiment,
        'probabilities': prob_dict
    }

# Display predictions on the test set
print("\nTesting the classifier")
for i in range(5):  # Show first 5 samples
    review = X_test.iloc[i]
    true_label = y_test.iloc[i]
    predicted_label = y_pred[i]
    positive_prob = tfidf_pipeline.predict_proba([review])[0][list(tfidf_pipeline.classes_).index('positive')]
    negative_prob = tfidf_pipeline.predict_proba([review])[0][list(tfidf_pipeline.classes_).index('negative')]
    
    print(f"Review #{i+1}: {review}")
    print(f"True Sentiment: {true_label}")
    print(f"Predicted Sentiment: {predicted_label}")
    print(f"Confidence - Positive: {positive_prob:.2f}, Negative: {negative_prob:.2f}")
