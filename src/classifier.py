import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report
from collections import Counter

import tf_keras as keras
from tf_keras import layers
import pickle
from src.config import CFG

class ResumeClassifier:
    def __init__(self, input_dim, num_classes):
        self.label_encoder = LabelEncoder()
        self.model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def train(self, X, y_raw, epochs: int = None, batch_size: int = None):
        epochs = epochs or CFG.CLASSIFIER_EPOCHS
        batch_size = batch_size or CFG.CLASSIFIER_BATCH_SIZE
        y = self.label_encoder.fit_transform(y_raw)
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42, stratify=y)

        # Compute class weights to handle imbalanced category distribution
        counts = Counter(y_train)
        total = len(y_train)
        class_weight = {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}

        history = self.model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=[keras.callbacks.EarlyStopping(
                patience=CFG.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            )]
        )
        y_pred = self.model.predict(X_test).argmax(axis=1)
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        return history, report
    
    def predict(self, X):
        """Returns (labels, confidences) where confidence is the max softmax probability."""
        probs = self.model.predict(X)
        idx = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        labels = self.label_encoder.inverse_transform(idx)
        return labels, confidences
    
    def save(self, model_path: str = None, encoder_path: str = None):
        model_path = model_path or CFG.MODEL_SAVE_PATH
        encoder_path = encoder_path or CFG.ENCODER_SAVE_PATH
        self.model.save(model_path)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
    def load(self, model_path: str = None, encoder_path: str = None):
        model_path = model_path or CFG.MODEL_SAVE_PATH
        encoder_path = encoder_path or CFG.ENCODER_SAVE_PATH
        self.model = keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)