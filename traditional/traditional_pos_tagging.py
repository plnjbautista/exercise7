import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

# Import the training data from training_data_pos.py
from training_data_pos import training_data

class HiddenMarkovModel:
    def __init__(self):
        # Initialize sets and probability dictionaries for Hidden Markov Model components
        self.states = set()
        self.vocab = set()
        self.initial_probabilities = defaultdict(float)
        self.trans_probs = defaultdict(lambda: defaultdict(float))
        self.emit_probs = defaultdict(lambda: defaultdict(float))

    def train(self, tagged_corpus):
        # Count frequencies needed to compute initial, transition, and emission probabilities
        tag_frequencies = defaultdict(int)
        transition_frequencies = defaultdict(lambda: defaultdict(int))
        emission_frequencies = defaultdict(lambda: defaultdict(int))
        start_frequencies = defaultdict(int)

        for tagged_sentence in tagged_corpus:
            prev_tag = None
            for i, (word, tag) in enumerate(tagged_sentence):
                self.states.add(tag)
                self.vocab.add(word)
                tag_frequencies[tag] += 1
                emission_frequencies[tag][word] += 1

                if i == 0:
                    start_frequencies[tag] += 1
                if prev_tag is not None:
                    transition_frequencies[prev_tag][tag] += 1
                prev_tag = tag

        # Calculate initial tag probabilities
        total_starts = sum(start_frequencies.values())
        for tag in self.states:
            self.initial_probabilities[tag] = start_frequencies[tag] / total_starts

        # Calculate transition probabilities: P(current_tag | previous_tag)
        for tag in self.states:
            for next_tag in self.states:
                self.trans_probs[tag][next_tag] = transition_frequencies[tag][next_tag] / tag_frequencies[tag]

        # Calculate emission probabilities: P(word | tag)
        for tag in self.states:
            for word in self.vocab:
                self.emit_probs[tag][word] = emission_frequencies[tag][word] / tag_frequencies[tag]

    def viterbi(self, sentence):
        viterbi_matrix = [{}]
        path = {}

        for tag in self.states:
            viterbi_matrix[0][tag] = self.initial_probabilities[tag] * self.emit_probs[tag].get(sentence[0], 1e-6)
            path[tag] = [tag]

        for t in range(1, len(sentence)):
            viterbi_matrix.append({})
            new_path = {}

            for current_tag in self.states:
                max_prob, best_previous_tag = max(
                    (viterbi_matrix[t - 1][prev_tag] * self.trans_probs[prev_tag].get(current_tag, 1e-6) *
                     self.emit_probs[current_tag].get(sentence[t], 1e-6), prev_tag)
                    for prev_tag in self.states
                )
                viterbi_matrix[t][current_tag] = max_prob
                new_path[current_tag] = path[best_previous_tag] + [current_tag]

            path = new_path

        max_final_prob, best_final_tag = max((viterbi_matrix[-1][tag], tag) for tag in self.states)
        return path[best_final_tag]

    def evaluate(self, test_sentences, true_tags):
        predicted_tags = []
        all_true_tags = []
        
        for sentence, true_tag in zip(test_sentences, true_tags):
            predicted_tag = self.viterbi(sentence)
            predicted_tags.extend(predicted_tag)
            all_true_tags.extend(true_tag)
        
        # Calculate Precision, Recall, F1-Score, and Accuracy
        precision, recall, f1, support = precision_recall_fscore_support(all_true_tags, predicted_tags, average='micro')
        accuracy = sum([1 if pred == true else 0 for pred, true in zip(predicted_tags, all_true_tags)]) / len(all_true_tags)
        
        return accuracy, precision, recall, f1


# Create and train the model using the imported training data
model = HiddenMarkovModel()
model.train(training_data)

# Test sentences
sentence1 = ['Manila', 'is', 'the', 'capital']  # Words are in vocab
sentence2 = ['Cebu', 'attracts', 'many', 'tourists']  # Words are in vocab
sentence3 = ['Manila', 'hosts', 'many', 'cultural']  # Words are in vocab

# Define the true tags for these test sentences
true_tags1 = ['NOUN', 'VERB', 'DET', 'NOUN']
true_tags2 = ['NOUN', 'VERB', 'ADJ', 'NOUN']
true_tags3 = ['NOUN', 'VERB', 'ADJ', 'ADJ']

test_sentences = [sentence1, sentence2, sentence3]
true_tags = [true_tags1, true_tags2, true_tags3]

# Evaluate the model
accuracy, precision, recall, f1 = model.evaluate(test_sentences, true_tags)

# Print out the evaluation results
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1)

# Testing the prediction on each sentence
tags1 = model.viterbi(sentence1)
tags2 = model.viterbi(sentence2)
tags3 = model.viterbi(sentence3)

print("Sentence 1:", sentence1)
print("Predicted:", tags1)
print("True tags:", true_tags1)

print("Sentence 2:", sentence2)
print("Predicted:", tags2)
print("True tags:", true_tags2)

print("Sentence 3:", sentence3)
print("Predicted:", tags3)
print("True tags:", true_tags3)