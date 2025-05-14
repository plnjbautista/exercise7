import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import string

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

# Define preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()] 
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# Load the dataset
df = pd.read_csv('../cafe_sentiment_dataset.csv')

# Check the structure of the dataset
print(df.head())

# Extracting the feature and target variables
X = df['review_text']
y = df['sentiment']

# Apply preprocessing to reviews
X = X.apply(preprocess_text)

# Convert labels to integers (0, 1 because it's binary classification)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Tokenizer to vectorize the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

# Convert text data to sequences of integers
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad the sequences to ensure uniform length
X_train_pad = pad_sequences(X_train_seq, padding='post', maxlen=100)
X_test_pad = pad_sequences(X_test_seq, padding='post', maxlen=100)

# Build the RNN model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128))
model.add(Masking(mask_value=0))  
model.add(SimpleRNN(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_pad, y_train, epochs=9, batch_size=64, validation_data=(X_test_pad, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy:{test_accuracy:.2f} or {test_accuracy*100:.2f}%")

# Make predictions on the test data
y_pred = model.predict(X_test_pad)
y_pred = (y_pred > 0.5).astype(int)

# Display classification report (precision, recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Additional Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nAdditional Metrics:")
print(f"Accuracy: {accuracy:.2f} or {accuracy*100:.2f}%")
print(f"Precision: {precision:.2f} or {precision*100:.2f}%")
print(f"Recall:{recall:.2f} or {recall*100:.2f}%")
print(f"F1-Score:{f1:.2f} or {f1*100:.2f}%")

# Display some sample predictions
print("Testing the classifier (0: Negative, 1: Positive)")
for i in range(5):
    print(f"Review: {X_test.iloc[i]}")
    print(f"True Sentiment: {y_test[i]}")
    print(f"Predicted Sentiment: {y_pred[i][0]}")
