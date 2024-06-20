import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import tensorflow as tf
import random
import json
import pickle

# Load intents from JSON file
with open("D:\\Chat bot\\English model\\intents2.json", encoding="utf-8") as file:
    data = json.load(file)

# Attempt to load preprocessed data from pickle file
try:
    with open("data2.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # Processing the intents
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    # Save processed data to pickle
    with open("data2.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Define bag_of_words function
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

# Define chat function
def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        bag = bag_of_words(inp, words)
        result = model.predict(np.array([bag]))[0]
        result_index = np.argmax(result)
        confidence = result[result_index]  # احتمالية الإجابة الأعلى

        if confidence > 0.25:  # تعيين حد للثقة هنا
            tag = labels[result_index]
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I'm not sure I understand. Could you rephrase your question?")

# Define and build the model
input_shape = len(words),
input_layer = tf.keras.Input(shape=input_shape)
model = tf.keras.Sequential([
    input_layer,
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(output[0]), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load model if exists or train
try:
    model.load_weights("model2_keras.weights.h5")
except:
    model.fit(np.array(training), np.array(output), epochs=1000, batch_size=8, verbose=1)
    model.save_weights("model2_keras.weights.h5")

# Start chatting
if __name__ == "__main__":
    chat()
