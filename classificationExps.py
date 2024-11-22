# -*- coding: utf-8 -*-
#plot confusion matrix with seaborn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import DataLoader

def plot_cm(cm):
    # Plot confusion matrix
    labels= [ "Discrim", "No Discrim"]
    #labels= [ "Facebook", "Twitter", "Instagram", "TikTok", "YouTube" ]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


dl = DataLoader('/content/drive/MyDrive/Works/Ting-Paolo-Berta/Datos.csv')

X_text, y_discrim = dl.get_data(input_type="text", label_column="DISCRIM")

# Get unique values and counts
unique_values, counts = dl.get_unique_values_and_counts(y_discrim)

print("Unique values:", unique_values)
print("Counts:", counts)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""### Fixed split test with Tf.Idf and NBClassifier"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=2)

# Create a Multinomial Naive Bayes classifier
classifier = MultinomialNB()

# Create a pipeline that combines the vectorizer and the classifier
model = make_pipeline(tfidf_vectorizer, classifier)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
F1=f1_score(y_test, y_pred, average="macro")
print("F1-score:", F1)

#m =confusion_matrix(y_test, y_pred)

#plot_cm(cm)

# Get the log probabilities of words for each class
log_probabilities = classifier.feature_log_prob_

# Calculate the difference in log probabilities for each word
word_diff = log_probabilities[1] - log_probabilities[0]  # Assuming binary classification, change the index accordingly

# Get the feature names (words) from the TfidfVectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

# Create a dictionary that maps words to their differences in log probabilities
word_diff_dict = dict(zip(feature_names, word_diff))

# Sort the words by their differences in log probabilities
sorted_word_diff = sorted(word_diff_dict.items(), key=lambda item: item[1], reverse=True)

# Print the sorted words and their differences
for word, diff in sorted_word_diff[:10]:
    print(f"Word: {word}\tLog Probability Difference: {diff}")

print()
# Sort the words by their differences in log probabilities
sorted_word_diff = sorted(word_diff_dict.items(), key=lambda item: item[1], reverse=False)
for word, diff in sorted_word_diff[:10]:
    print(f"Word: {word}\tLog Probability Difference: {diff}")

import xgboost as xgb

X_mtrain = tfidf_vectorizer.fit_transform(X_train)
X_mtest = tfidf_vectorizer.fit_transform(X_test)

dtrain = xgb.DMatrix(X_mtrain, label=y_train)
dtest = xgb.DMatrix(X_mtest, label=y_test)
evallist = [(dtrain, 'train'), (dtest, 'eval')]

#param = {'max_depth': 5, 'eta': 1, 'objective': 'binary:logistic'}
param = {'max_depth': 4, 'eta': 1, 'objective': 'multi:softmax'}
#param['eval_metric'] = 'auc'
param['nthread'] = 4
param['eval_metric'] = 'mlogloss'
param['num_class'] = 4 #2 for binary
#param['scale_pos_weight'] = 0.33

#param['eval_metric'] = ['auc', 'ams@0']

bst = xgb.train(param, dtrain, 20, evallist, early_stopping_rounds=20)

y_pred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

plot_cm(cm)

"""### 10-fold cross validation"""

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

# Create a 10-fold cross-validation strategy
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Create a pipeline that combines the vectorizer and the classifier
model = make_pipeline(tfidf_vectorizer, classifier)

# Perform 10-fold cross-validation and calculate accuracy scores
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

# Calculate the mean accuracy and standard deviation
mean_accuracy = scores.mean()
std_accuracy = scores.std()

print("Mean Accuracy:", mean_accuracy)
print("Standard Deviation:", std_accuracy)

"""### BERT classifier"""

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name) #, do_lower_case=True ?
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(unique_values)+1)
model.to(device)

# Tokenize your text data
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in X]

# Pad and truncate tokenized texts
max_len = max(len(tokens) for tokens in tokenized_texts)
max_seq_length = 512
truncated_texts = [ tokens[:max_seq_length] for tokens in tokenized_texts]

input_ids = [tokens + [0] * (max_seq_length - len(tokens)) for tokens in truncated_texts]

# Create attention masks to indicate which tokens are padding
attention_masks = [[int(token_id > 0) for token_id in input_id] for input_id in input_ids]

# Convert to PyTorch tensors
input_ids = torch.tensor(input_ids).to(device)
labels = torch.tensor(y).to(device)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(input_ids, labels, test_size=0.2, random_state=42)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, test_size=0.2, random_state=42)

train_masks= torch.tensor(train_masks).to(device)
validation_masks= torch.tensor(validation_masks).to(device)

# Create data loaders
train_data = TensorDataset(X_train, train_masks, y_train)
train_loader = DataLoader(train_data, batch_size=16)

# Fine-tune BERT model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
for epoch in range(20):
    for batch in train_loader:
        inputs, mask, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

model_save_directory = "./BERT_pretrained_Discrim"
tokenizer.save_pretrained(model_save_directory)
model.save_pretrained(model_save_directory)

tokenizer = BertTokenizer.from_pretrained("./BERT_pretrained_Discrim", local_files_only=True)
model = BertForSequenceClassification.from_pretrained("./BERT_pretrained_Discrim", num_labels=len(unique_values)+1)

# Tokenize your text data
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in X]

# Pad and truncate tokenized texts
max_len = max(len(tokens) for tokens in tokenized_texts)
max_seq_length = 512
truncated_texts = [ tokens[:max_seq_length] for tokens in tokenized_texts]

input_ids = [tokens + [0] * (max_seq_length - len(tokens)) for tokens in truncated_texts]

# Create attention masks to indicate which tokens are padding
attention_masks = [[int(token_id > 0) for token_id in input_id] for input_id in input_ids]

# Convert to PyTorch tensors
input_ids = torch.tensor(input_ids)
labels = torch.tensor(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(input_ids, labels, test_size=0.2, random_state=42)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, test_size=0.2, random_state=42)

train_masks= torch.tensor(train_masks)
validation_masks= torch.tensor(validation_masks)


# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test, attention_mask=validation_masks)
    predicted_labels = predictions.logits.argmax(dim=1)
accuracy = accuracy_score(y_test, predicted_labels)
print("Accuracy:", accuracy)

F1 = f1_score(y_test, predicted_labels, average="macro")
print("F1-score:", F1)

print(predicted_labels)
