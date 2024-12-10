import pandas as pd
import string
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
nltk.download('stopwords')

dataset = pd.read_csv('data.csv')

def clean_text(line):
    line = line.lower()
    line = re.sub(r'\d+', '', line)
    line = re.sub(r'[^a-zA-Z0-9\s]', '', line)
    translator = str.maketrans('', '', string.punctuation)
    line = line.translate(translator)
    words = [word for word in line.split() if word not in stopwords.words('english')]
    return ' '.join(words)

dataset['Sentence'] = dataset['Sentence'].apply(clean_text)

label_encoder = LabelEncoder()
dataset['Sentiment'] = label_encoder.fit_transform(dataset['Sentiment'])
encoded_classes = list(label_encoder.classes_)

tokenizer = Tokenizer(oov_token='<unk>', num_words=2500)
tokenizer.fit_on_texts(dataset['Sentence'].values)
data_x = tokenizer.texts_to_sequences(dataset['Sentence'].values)
data_x = pad_sequences(data_x, maxlen=42, padding='post', truncating='post')

data_y = pd.get_dummies(dataset['Sentiment']).values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=0, stratify=data_y)

model = Sequential()
model.add(Embedding(input_dim=2500, output_dim=50, input_length=42))
model.add(Dropout(0.25))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(200))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))  # Adjust for the number of sentiment classes
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=30, epochs=10, validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test)
print(f"Test loss: {score}")
print(f"Test accuracy: {acc}")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Batch 30 Training vs Validation Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Batch 30 Training vs Validation Loss')
plt.legend()
plt.show()

# Predict on test data
predictions = model.predict(x_test)
predicted_classes = label_encoder.inverse_transform(predictions.argmax(axis=1))
