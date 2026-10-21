
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

tf.keras.utils.set_random_seed(42)

pairs = [
    ("i am happy", "मैं खुश हूँ"),
    ("how are you", "आप कैसे हैं"),
    ("what is your name", "आपका नाम क्या है"),
    ("my name is aditya", "मेरा नाम आदित्य है"),
    ("i love india", "मुझे भारत से प्यार है"),
    ("good morning", "सुप्रभात"),
    ("thank you", "धन्यवाद"),
    ("where are you going", "आप कहाँ जा रहे हैं"),
    ("i am a student", "मैं एक छात्र हूँ"),
    ("this is my book", "यह मेरी किताब है"),
    ("he is my friend", "वह मेरा दोस्त है"),
    ("she is a teacher", "वह एक शिक्षिका है"),
    ("i like tea", "मुझे चाय पसंद है"),
    ("the water is cold", "पानी ठंडा है"),
    ("open the door", "दरवाज़ा खोलो"),
    ("come here", "यहाँ आओ"),
    ("i am hungry", "मुझे भूख लगी है"),
    ("we are friends", "हम दोस्त हैं"),
    ("what is this", "यह क्या है"),
    ("i live in nagpur", "मैं नागपुर में रहता हूँ"),
    ("i study engineering", "मैं इंजीनियरिंग पढ़ता हूँ"),
    ("see you tomorrow", "कल मिलते हैं"),
    ("have a nice day", "आपका दिन शुभ हो"),
    ("i am learning ai", "मैं एआई सीख रहा हूँ"),
    ("the weather is nice", "मौसम अच्छा है")
]

src_texts = [src for src, tgt in pairs]
tgt_texts = ["<start> " + tgt + " <end>" for src, tgt in pairs]


def build_vocab(sentences):
    words = sorted(set(" ".join(sentences).split()))
    word_to_index = {word: i + 1 for i, word in enumerate(words)}
    index_to_word = {i: word for word, i in word_to_index.items()}
    return word_to_index, index_to_word


src_w2i, _ = build_vocab(src_texts)
tgt_w2i, tgt_i2w = build_vocab(tgt_texts)

src_vocab = len(src_w2i) + 1
tgt_vocab = len(tgt_w2i) + 1


def encode(sentences, word_to_index, max_len):
    data = np.zeros((len(sentences), max_len), dtype="int32")

    for row, sentence in enumerate(sentences):
        for col, word in enumerate(sentence.split()):
            data[row, col] = word_to_index[word]

    return data


max_src = max(len(s.split()) for s in src_texts)
max_tgt = max(len(s.split()) for s in tgt_texts)

enc_X = encode(src_texts, src_w2i, max_src)
dec_full = encode(tgt_texts, tgt_w2i, max_tgt)

dec_X = dec_full[:, :-1]
dec_y = dec_full[:, 1:]


# Encoder
enc_in = Input(shape=(None,))
enc_emb = Embedding(src_vocab, 64)(enc_in)

_, h, c = LSTM(128, return_state=True)(enc_emb)


# Decoder
dec_in = Input(shape=(None,))
dec_emb_layer = Embedding(tgt_vocab, 64)
dec_lstm = LSTM(128, return_sequences=True, return_state=True)
dec_dense = Dense(tgt_vocab, activation="softmax")

dec_seq, _, _ = dec_lstm(
    dec_emb_layer(dec_in),
    initial_state=[h, c]
)

output = dec_dense(dec_seq)

model = Model([enc_in, dec_in], output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    [enc_X, dec_X],
    dec_y,
    epochs=300,
    batch_size=8,
    verbose=0
)

print("Training Accuracy:", round(history.history["accuracy"][-1], 2))


# Inference models
enc_model = Model(enc_in, [h, c])

state_h = Input(shape=(128,))
state_c = Input(shape=(128,))

dec_seq2, h2, c2 = dec_lstm(
    dec_emb_layer(dec_in),
    initial_state=[state_h, state_c]
)

dec_model = Model(
    [dec_in, state_h, state_c],
    [dec_dense(dec_seq2), h2, c2]
)


# Translation function
def translate(sentence):
    tokens = [
        src_w2i[word]
        for word in sentence.lower().split()
        if word in src_w2i
    ]

    seq = np.zeros((1, max_src), dtype="int32")
    seq[0, :len(tokens)] = tokens

    h_s, c_s = enc_model.predict(seq, verbose=0)
    word_id = tgt_w2i["<start>"]
    result = []

    for _ in range(max_tgt):
        probs, h_s, c_s = dec_model.predict(
            [np.array([[word_id]]), h_s, c_s],
            verbose=0
        )

        word_id = int(np.argmax(probs[0, -1]))
        word = tgt_i2w.get(word_id, "")

        if word == "<end>" or word == "":
            break

        result.append(word)

    return " ".join(result)


# Test translations
test_sentences = [
    "how are you",
    "i am happy",
    "thank you",
    "i live in nagpur",
    "the weather is nice"
]

for sentence in test_sentences:
    print(sentence, "->", translate(sentence))
