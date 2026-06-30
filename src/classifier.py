import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report

from tensorflow import keras
from tensorflow.keras import layers
import pickle

class ResumeClassifier:
    def __init__(self, input_dim, num_classes):
        self.label_encoder = LabelEncoder()
        self.model = keras.Sequential([
            layers.Input(shape = (input_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def train(self, X, y_raw, epochs=50, batch_size=32):
        y = self.label_encoder.fit_transform(y_raw)
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42, stratify=y) 
        
        history=self.model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )
        y_pred = self.model.predict(X_test).argmax(axis=1)
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        return history, report
    
    def predict(self, X):
        probs = self.model.predict(X)
        idx = probs.argmax(axis=1)
        return self.label_encoder.inverse_transform(idx)
    
    def save(self, model_path, encoder_path):
        self.model.save(model_path)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
    def load(self, model_path, encoder_path):
        self.model = keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)