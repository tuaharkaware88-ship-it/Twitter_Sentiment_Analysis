from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
from Tokenizer import padding_text
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Build Model:

def rnn_model(vocab_size, out_dim, max_len):

    print("Building RNN model...")

    X_train_padded  = np.load(r"C:\Sentiment_Analysis_Project\data\padded_data\X_train_padded.npy")
    X_test_padded = np.load(r"C:\Sentiment_Analysis_Project\data\padded_data\X_test_padded.npy")
     
    y_train = pd.read_csv(r"C:\Sentiment_Analysis_Project\data\processed\y_train.csv")["target"]
    y_test = pd.read_csv(r"C:\Sentiment_Analysis_Project\data\processed\y_test.csv")["target"]

    print("Creating Sequential model...")

    model=Sequential()    

    model.add(Embedding(input_dim=vocab_size, output_dim=out_dim, input_length=max_len))
    model.add(Dropout(0.3))

    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation="sigmoid"))         # Output layer

    print("Compiling model...")

    # Backpropogation
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    print("Model compiled.")
    
    model.build(input_shape=(None, max_len))

    print("Model built. Summary:")
    print(model.summary())  

    # Train the model:

    history = model.fit(
        X_train_padded, y_train,
        validation_data=(X_test_padded, y_test),
        epochs=2,
        batch_size=128,
        verbose=1)
    
    print(history.history)

    model_path = r"C:\Sentiment_Analysis_Project\models\rnn\rnn_model.h5"       # Keras model not compatable with joblib and pickel
    model.save(model_path)
    print(f"✅ RNN model saved to: {model_path}")
    
    plot_training_history(history)

    return model, history


def plot_training_history(history):
    """Plot accuracy and loss curves."""
    print("Plotting training history...")

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model, history = rnn_model(vocab_size=10000, out_dim=100, max_len=100)







    