import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ============================================================
# 1. SET RANDOM SEEDS
# ============================================================
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 2. TRAINING TEXT
# ============================================================
text = """
i am going home
i am going to school
i am going to college
i want to go home
i want to eat food
i want to play football
i want to learn python
i like to play football
i like to eat pizza
i like to learn programming
i love to play cricket
i love to eat pizza
i love programming
i am learning python
i am learning machine learning
i am learning deep learning
machine learning is interesting
deep learning is very powerful
python is easy to learn
python is a programming language
this is a good day
this is a beautiful day
this is my home
this is my college
how are you
how are you doing
what are you doing
what is your name
where are you going
where are you from
can you help me
can you help me with python
please help me
please give me some information
thank you for your help
have a nice day
see you tomorrow
good morning everyone
good night everyone
"""

# ============================================================
# 3. TOKENIZATION
# ============================================================
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
print("Vocabulary size:", total_words)

# ============================================================
# 4. CREATE N-GRAM SEQUENCES
# ============================================================
input_sequences = []
sentences = text.strip().split("\n")

for sentence in sentences:
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# ============================================================
# 5. PAD SEQUENCES
# ============================================================
max_sequence_len = max(len(seq) for seq in input_sequences)
print("Maximum sequence length:", max_sequence_len)

input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
)

# ============================================================
# 6. CREATE X AND y
# ============================================================
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# 7. BUILD LSTM MODEL
# ============================================================
model = Sequential([
    Embedding(total_words, 64, input_length=max_sequence_len-1),
    LSTM(128),
    Dense(total_words, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print("\nModel Summary:\n")
model.summary()

# ============================================================
# 8. EARLY STOPPING
# ============================================================
early_stopping = EarlyStopping(monitor="loss", patience=30, restore_best_weights=True)

# ============================================================
# 9. TRAIN MODEL
# ============================================================
print("\n====================================")
print("       TRAINING MODEL")
print("====================================\n")

history = model.fit(X, y, epochs=500, verbose=1, callbacks=[early_stopping])
print("\nTraining completed!")

# ============================================================
# 10. PLOT TRAINING HISTORY
# ============================================================
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.title("Training Accuracy & Loss")
plt.show()

# ============================================================
# 11. REVERSE WORD INDEX
# ============================================================
reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}

# ============================================================
# 12. NEXT WORD PREDICTION FUNCTION
# ============================================================
def predict_next_words(seed_text, number_of_words=5):
    seed_text = seed_text.lower().strip()
    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    if len(token_list) == 0:
        return []

    token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")

    probabilities = model.predict(token_list, verbose=0)[0]
    top_indices = np.argsort(probabilities)[-number_of_words:][::-1]

    predictions = []
    for index in top_indices:
        word = reverse_word_index.get(index)
        if word:
            predictions.append((word, round(float(probabilities[index]) * 100, 2)))
    return predictions

# ============================================================
# 13. INTERACTIVE LOOP
# ============================================================
print("\n====================================")
print("     NEXT WORD PREDICTION SYSTEM")
print("====================================")
print("\nType a sentence and press Enter.")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("Enter text: ")
    if user_input.lower().strip() == "exit":
        print("\nProgram ended.")
        break

    if user_input.strip() == "":
        print("Please enter some text.\n")
        continue

    predictions = predict_next_words(user_input, number_of_words=5)
    print("\nSuggested next words:")
    if not predictions:
        print("No prediction available.")
    else:
        for i, (word, prob) in enumerate(predictions, start=1):
            print(f"{i}. {word} ({prob}%)")
    print()
