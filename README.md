# AI_Mental_therapist_Chatbot
!pip install tensorflow
!pip install flask
!pip install requests
!pip install tflearn
!pip install --upgrade tflearn
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
# Load intents from JSON data
with open("/content/data.json") as json_data:
    intents = json.load(json_data)
# Extract data from intents
training_data = []
labels = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        training_data.append(pattern)
        labels.append(intent["tag"])
# Encode labels as integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_data)
word_index = tokenizer.word_index
# Convert text data to sequences of integers
sequences = tokenizer.texts_to_sequences(training_data)
# Pad sequences to ensure uniform length
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
# Define the model architecture
vocab_size = len(word_index) + 1
embedding_dim = 128
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),
    LSTM(128, return_sequences=True),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dense(len(set(labels)), activation='softmax')
])
# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
model.fit(padded_sequences, np.array(encoded_labels), epochs=130)
2
# Save the model
model.save("chatbot_model.h5")
# Install the library for integrating the Telegram Bot API
!pip install pyTelegramBotAPI
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import telebot
# Load intents from JSON data
with open("/content/data.json") as json_data:
    intents = json.load(json_data)
# Load the trained model
model = load_model("chatbot_model.h5")
# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([pattern for intent in intents["intents"] for pattern in intent["patterns"]])
word_index = tokenizer.word_index
# Function to preprocess input text
def preprocess_input(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    return padded_sequence

# Function to predict intent
def predict_intent(text):
    preprocessed_text = preprocess_input(text)
    predictions = model.predict(preprocessed_text)
    predicted_label_index = np.argmax(predictions)
    predicted_label = label_encoder.inverse_transform([predicted_label_index])
return predicted_label[0]
# Set up Telegram bot
bot = telebot.TeleBot("7275296661:AAFSh5AGIxgqfLGnOj0yBH18ya0E2fX-0qw")
# Define message response function
def generate_response(message):
    intent = predict_intent(message.text)
    for i in intents["intents"]:
        if i['tag'] == intent:
            return np.random.choice(i['responses'])
    return "Sorry, I'm not sure how to respond to that."
# Set up message handler
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    bot.send_message(message.chat.id, generate_response(message))
# Start bot
bot.polling()
