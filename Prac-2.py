# Install required libraries
!pip install -q transformers[torch] datasets scikit-learn seaborn matplotlib

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# -------------------------------
# 1. Prepare Dataset
# -------------------------------
data = {
    'text': [
        'The football team won the championship.', 'Tennis tournament finals are today.',
        'New record in athletics.', 'Basketball playoffs started.', 'Cricket world cup highlights.',
        'The government announced new policies.', 'Parliament passed the bill.',
        'Election results are out.', 'The president signed the treaty.', 'Protests against the new law.',
        'AI breakthrough in computing.', 'New smartphone launched with OLED.',
        'Cybersecurity threats are rising.', 'Startup unveils quantum chip.', 'Software update released.'
    ] * 4,  # repeat to increase dataset size
    'label': [0]*20 + [1]*20 + [2]*20  # 0=Sports, 1=Politics, 2=Technology
}

df = pd.DataFrame(data)

# Train-validation split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Convert to HuggingFace Dataset
ds = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'validation': Dataset.from_pandas(val_df)
})

# -------------------------------
# 2. Load Tokenizer and Tokenize
# -------------------------------
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

tokenized_ds = ds.map(tokenize, batched=True)
tokenized_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# -------------------------------
# 3. Load Model
# -------------------------------
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# -------------------------------
# 4. Define Metrics
# -------------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

# -------------------------------
# 5. Training Arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    report_to='none'
)

# -------------------------------
# 6. Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['validation'],
    compute_metrics=compute_metrics,
)

# -------------------------------
# 7. Train
# -------------------------------
trainer.train()

# -------------------------------
# 8. Evaluate
# -------------------------------
metrics = trainer.evaluate()
print(f"\nValidation Metrics: {metrics}")

# -------------------------------
# 9. Classification Report
# -------------------------------
predictions = trainer.predict(tokenized_ds['validation'])
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

target_names = ['Sports', 'Politics', 'Technology']
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# -------------------------------
# 10. Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for News Classification')
plt.show()
