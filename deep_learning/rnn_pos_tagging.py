import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from collections import Counter

# Training data
from training_data_pos import training_data

# Create word-to-index and tag-to-index mappings
word_to_idx = {'<PAD>': 0}
tag_to_idx = {'<PAD>': 0}
word_idx = 1
tag_idx = 1

for sentence in training_data:
    for word, tag in sentence:
        if word not in word_to_idx:
            word_to_idx[word] = word_idx
            word_idx += 1
        if tag not in tag_to_idx:
            tag_to_idx[tag] = tag_idx
            tag_idx += 1

# Reverse dictionaries for later usage
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}

# Prepare input sequences and labels
X_train = [[word_to_idx[word] for word, tag in sentence] for sentence in training_data]
y_train = [[tag_to_idx[tag] for word, tag in sentence] for sentence in training_data]

# Pad sequences to have the same length
max_len = max(len(sentence) for sentence in X_train)

def pad_sequences(sequences, max_len):
    return [seq + [0] * (max_len - len(seq)) for seq in sequences]

X_train_padded = pad_sequences(X_train, max_len)
y_train_padded = pad_sequences(y_train, max_len)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_padded, dtype=torch.long)
y_train_tensor = torch.tensor(y_train_padded, dtype=torch.long)

# Define the RNN Model
class POS_RNN(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=200, hidden_dim=256):
        super(POS_RNN, self).__init__()
        
        # Define layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)  # RNN layer
        self.fc = nn.Linear(hidden_dim, tagset_size)  # Fully connected layer

    def forward(self, x):
        embedded = self.embedding(x)  # Get embeddings for words
        output, _ = self.rnn(embedded)  # Get RNN outputs
        output = self.fc(output)  # Get final tag predictions
        return output

# Create the model
model = POS_RNN(vocab_size=len(word_to_idx), tagset_size=len(tag_to_idx))

# Hyperparameters
epochs = 50
learning_rate = 0.001

# Flatten your label list to count class frequencies
all_labels = [tag for sent in y_train for tag in sent]
label_counts = Counter(all_labels)

# Create a list of counts aligned with tag indices (e.g., [count for 0, 1, 2...])
class_counts = [label_counts.get(i, 1) for i in range(len(tag_to_idx))]  # avoid div by 0
weights = torch.tensor(1.0 / np.array(class_counts), dtype=torch.float)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()  # Clear previous gradients
    
    # Forward pass
    outputs = model(X_train_tensor)
    
    # Reshape the output and labels for CrossEntropy
    outputs = outputs.view(-1, len(tag_to_idx))  # Flatten the output
    labels = y_train_tensor.view(-1)  # Flatten the labels

    # Calculate loss
    loss = criterion(outputs, labels)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print loss every few epochs
    if (epoch + 1) % 2 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Test sentences
sentence1 = ['Manila', 'is', 'the', 'capital']  # Words are in vocab
sentence2 = ['Cebu', 'attracts', 'many', 'tourists']  # Words are in vocab
sentence3 = ['Manila', 'hosts', 'many', 'cultural']  # Words are in vocab

# Define the true tags for these test sentences
true_tags1 = ['NOUN', 'VERB', 'DET', 'NOUN']
true_tags2 = ['NOUN', 'VERB', 'ADJ', 'NOUN']
true_tags3 = ['NOUN', 'VERB', 'ADJ', 'ADJ']

# List of test sentences and true tags
test_sentences = [sentence1, sentence2, sentence3]
true_tags = [true_tags1, true_tags2, true_tags3]

# Convert test sentences to indices
def sentence_to_indices(sentence):
    return [word_to_idx.get(word, 0) for word in sentence]

test_sentences_idx = [sentence_to_indices(sentence) for sentence in test_sentences]
test_sentences_tensor = torch.tensor(pad_sequences(test_sentences_idx, max_len), dtype=torch.long)

# Evaluate the model
model.eval()
all_predictions = []
all_true_labels = []

print("\n--- Predictions per Sentence ---")
with torch.no_grad():
    for i, sentence_tensor in enumerate(test_sentences_tensor):
        sentence_words = test_sentences[i]
        true_tags_sentence = true_tags[i]

        # Get the model's predictions
        outputs = model(sentence_tensor.unsqueeze(0))  # Add batch dimension
        _, predicted_tag_indices = torch.max(outputs, dim=2)  # Get predicted tag indices
        predicted_tag_indices = predicted_tag_indices.squeeze(0).tolist()  # Remove batch dimension

        # Convert indices to tags
        predicted_tags = [idx_to_tag[idx] for idx in predicted_tag_indices[:len(sentence_words)]]

        # Display
        print(f"\nSentence {i+1}: {' '.join(sentence_words)}")
        print(f"Predicted Tags: {predicted_tags}")
        print(f"True Tags:      {true_tags_sentence}")

        # Accumulate for overall evaluation
        all_predictions.extend(predicted_tags)
        all_true_labels.extend(true_tags_sentence)

# Evaluation metrics
accuracy = accuracy_score(all_true_labels, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_true_labels, all_predictions, average='weighted', zero_division=0
)

print("\n--- Evaluation Metrics ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Detailed classification report
report = classification_report(all_true_labels, all_predictions, zero_division=0, digits=4, output_dict=True)
print("\n--- Classification Report ---\n")
import pandas as pd
df_report = pd.DataFrame(report).transpose()
print(df_report.sort_values('support', ascending=False))

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(all_true_labels, all_predictions, labels=list(tag_to_idx.keys()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(tag_to_idx.keys()))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.show()

from sklearn.metrics import classification_report
print(classification_report(all_true_labels, all_predictions))
