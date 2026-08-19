import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# ============================================================
# 1. TRAINING DATA
# ============================================================

english_sentences = [
    "hello",
    "how are you",
    "i am fine",
    "good morning",
    "thank you",
    "what is your name",
    "i am a student",
    "i love india"
]

hindi_sentences = [
    "नमस्ते",
    "आप कैसे हैं",
    "मैं ठीक हूँ",
    "सुप्रभात",
    "धन्यवाद",
    "आपका नाम क्या है",
    "मैं एक विद्यार्थी हूँ",
    "मुझे भारत से प्यार है"
]

# ============================================================
# 2. CREATE VOCABULARY
# ============================================================

eng_words = set()
hin_words = set()

for sentence in english_sentences:
    eng_words.update(sentence.split())

for sentence in hindi_sentences:
    hin_words.update(sentence.split())

# +1 because 0 is reserved for padding/start token
eng_token = {word: i + 1 for i, word in enumerate(eng_words)}
hin_token = {word: i + 1 for i, word in enumerate(hin_words)}

print("English Vocabulary:")
print(eng_token)

print("\nHindi Vocabulary:")
print(hin_token)

# ============================================================
# 3. CONVERT WORDS INTO NUMBERS
# ============================================================

encoder_input = []
decoder_input = []
decoder_output = []

for eng, hin in zip(english_sentences, hindi_sentences):

    # English sentence -> numbers
    enc = [eng_token[word] for word in eng.split()]

    # Hindi sentence -> numbers
    dec = [hin_token[word] for word in hin.split()]

    encoder_input.append(enc)

    # Decoder input starts with 0
    # Example:
    # Hindi:  मैं ठीक हूँ
    # Input:  0  मैं ठीक
    decoder_input.append([0] + dec[:-1])

    # Expected decoder output
    # Example:
    # Output: मैं ठीक हूँ
    decoder_output.append(dec)

# ============================================================
# 4. PADDING
# ============================================================

max_eng_len = max(len(x) for x in encoder_input)
max_hin_len = max(len(x) for x in decoder_input)

encoder_input = tf.keras.utils.pad_sequences(
    encoder_input,
    maxlen=max_eng_len,
    padding="post"
)

decoder_input = tf.keras.utils.pad_sequences(
    decoder_input,
    maxlen=max_hin_len,
    padding="post"
)

decoder_output = tf.keras.utils.pad_sequences(
    decoder_output,
    maxlen=max_hin_len,
    padding="post"
)

print("\nEncoder Input:")
print(encoder_input)

print("\nDecoder Input:")
print(decoder_input)

print("\nDecoder Output:")
print(decoder_output)

# ============================================================
# 5. ENCODER
# ============================================================

encoder_inputs = Input(
    shape=(max_eng_len,),
    name="encoder_inputs"
)

# Embedding layer
encoder_embedding_layer = Embedding(
    input_dim=len(eng_token) + 1,
    output_dim=64,
    name="encoder_embedding_layer"
)

encoder_embedding_output = encoder_embedding_layer(
    encoder_inputs
)

# LSTM layer
encoder_lstm_layer = LSTM(
    128,
    return_state=True,
    name="encoder_lstm_layer"
)

encoder_outputs, state_h, state_c = encoder_lstm_layer(
    encoder_embedding_output
)

# ============================================================
# 6. DECODER
# ============================================================

decoder_inputs = Input(
    shape=(max_hin_len,),
    name="decoder_inputs"
)

# Hindi Embedding
decoder_embedding_layer = Embedding(
    input_dim=len(hin_token) + 1,
    output_dim=64,
    name="decoder_embedding_layer"
)

decoder_embedding_output = decoder_embedding_layer(
    decoder_inputs
)

# Decoder LSTM
decoder_lstm_layer = LSTM(
    128,
    return_sequences=True,
    return_state=True,
    name="decoder_lstm_layer"
)

decoder_outputs, _, _ = decoder_lstm_layer(
    decoder_embedding_output,
    initial_state=[state_h, state_c]
)

# Dense output layer
decoder_dense_layer = Dense(
    len(hin_token) + 1,
    activation="softmax",
    name="decoder_dense_layer"
)

outputs = decoder_dense_layer(
    decoder_outputs
)

# ============================================================
# 7. CREATE MODEL
# ============================================================

model = Model(
    [encoder_inputs, decoder_inputs],
    outputs
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n================ MODEL SUMMARY ================\n")
model.summary()

# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\n================ TRAINING ================\n")

model.fit(
    [encoder_input, decoder_input],
    decoder_output,
    batch_size=2,
    epochs=200,
    verbose=1
)

print("\nEnglish to Hindi model trained successfully!")

# ============================================================
# 9. ENCODER INFERENCE MODEL
# ============================================================

encoder_model_inf = Model(
    encoder_inputs,
    [state_h, state_c]
)

# ============================================================
# 10. DECODER INFERENCE MODEL
# ============================================================

# Input states
decoder_state_input_h = Input(
    shape=(128,),
    name="decoder_state_input_h"
)

decoder_state_input_c = Input(
    shape=(128,),
    name="decoder_state_input_c"
)

decoder_states_inputs = [
    decoder_state_input_h,
    decoder_state_input_c
]

# Input one word at a time
decoder_input_inference = Input(
    shape=(1,),
    name="decoder_input_inference"
)

# Reuse Hindi embedding layer
decoder_embedding_inf = decoder_embedding_layer(
    decoder_input_inference
)

# Reuse LSTM
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm_layer(
    decoder_embedding_inf,
    initial_state=decoder_states_inputs
)

decoder_states_inf = [
    state_h_inf,
    state_c_inf
]

# Reuse Dense layer
decoder_outputs_inf = decoder_dense_layer(
    decoder_outputs_inf
)

# Create decoder inference model
decoder_model_inf = Model(
    [decoder_input_inference] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf
)

# ============================================================
# 11. REVERSE VOCABULARY
# ============================================================

reverse_eng_word_index = {
    i: word for word, i in eng_token.items()
}

reverse_hin_word_index = {
    i: word for word, i in hin_token.items()
}

# 0 = start/padding token
reverse_hin_word_index[0] = ""

# ============================================================
# 12. TRANSLATION FUNCTION
# ============================================================

def decode_sequence(input_sentence):

    # Convert English words into numbers
    input_seq = [
        eng_token.get(word, 0)
        for word in input_sentence.lower().split()
    ]

    # Padding
    input_seq = tf.keras.utils.pad_sequences(
        [input_seq],
        maxlen=max_eng_len,
        padding="post"
    )

    # Get encoder states
    states_value = encoder_model_inf.predict(
        input_seq,
        verbose=0
    )

    decoded_sentence = []

    # Start token = 0
    target_seq = np.zeros(
        (1, 1),
        dtype="int32"
    )

    stop_condition = False

    while not stop_condition:

        output_tokens, h, c = decoder_model_inf.predict(
            [target_seq] + states_value,
            verbose=0
        )

        # Select word with highest probability
        sampled_token_index = np.argmax(
            output_tokens[0, -1, :]
        )

        sampled_word = reverse_hin_word_index.get(
            sampled_token_index,
            ""
        )

        # Stop if padding/unknown token
        if sampled_token_index == 0 or sampled_word == "":
            break

        decoded_sentence.append(
            sampled_word
        )

        # Stop at maximum length
        if len(decoded_sentence) >= max_hin_len:
            stop_condition = True

        # Send predicted word back to decoder
        target_seq = np.zeros(
            (1, 1),
            dtype="int32"
        )

        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return " ".join(decoded_sentence)


# ============================================================
# 13. TEST TRAINING SENTENCES
# ============================================================

print("\n==============================================")
print("TRANSLATIONS FROM TRAINING DATA")
print("==============================================\n")

for i in range(len(english_sentences)):

    input_text = english_sentences[i]

    translated_text = decode_sequence(
        input_text
    )

    print("Input English   :", input_text)
    print("Expected Hindi  :", hindi_sentences[i])
    print("Predicted Hindi :", translated_text)
    print("----------------------------------------------")


# ============================================================
# 14. TEST NEW SENTENCE
# ============================================================

print("\n==============================================")
print("TESTING NEW SENTENCE")
print("==============================================\n")

new_english_sentence = "i am good"

predicted_new = decode_sequence(
    new_english_sentence
)

print("Input English   :", new_english_sentence)
print("Predicted Hindi :", predicted_new) 
