import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# Training data
training_data = [
    [('Manila', 'NOUN'), ('is', 'VERB'), ('the', 'DET'), ('capital', 'NOUN'), ('of', 'NOUN'), ('Philippines', 'NOUN')],
    [('Cebu', 'NOUN'), ('attracts', 'VERB'), ('many', 'ADJ'), ('tourists', 'NOUN'), ('annually', 'ADV')],
    [('The', 'DET'), ('Filipino', 'ADJ'), ('cuisine', 'NOUN'), ('features', 'VERB'), ('tropical', 'ADJ'), ('fruits', 'NOUN')],
    [('Boracay', 'NOUN'), ('has', 'VERB'), ('beautiful', 'ADJ'), ('white', 'ADJ'), ('beaches', 'NOUN')],
    [('Tagalog', 'NOUN'), ('is', 'VERB'), ('an', 'DET'), ('official', 'ADJ'), ('language', 'NOUN')],
    [('President', 'NOUN'), ('Marcos', 'NOUN'), ('governs', 'VERB'), ('the', 'DET'), ('country', 'NOUN')],
    [('Jeepneys', 'NOUN'), ('provide', 'VERB'), ('unique', 'ADJ'), ('transportation', 'NOUN'), ('options', 'NOUN')],
    [('Manny', 'NOUN'), ('Pacquiao', 'NOUN'), ('represented', 'VERB'), ('Filipino', 'ADJ'), ('boxing', 'NOUN'), ('internationally', 'ADV')]
]

# Create word-to-index and tag-to-index mappings
word_to_idx = {}
tag_to_idx = {}
word_idx = 0
tag_idx = 0

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
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128):
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
epochs = 20
learning_rate = 0.1

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index (0)
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
report = classification_report(all_true_labels, all_predictions, zero_division=0)
print("\n--- Classification Report ---\n")
print(report)
