import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(units=64, input_dim=20, activation='relu'))  # Primo strato nascosto con 64 neuroni e attivazione ReLU
model.add(Dense(units=32, activation='relu'))  # Secondo strato nascosto con 32 neuroni e attivazione ReLU
model.add(Dense(units=1, activation='sigmoid'))  # Strato di output con 1 neurone e attivazione sigmoide

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

new_data = np.random.rand(5, 20)  # Esempio di nuovi dati casuali
predictions = model.predict(new_data)
print(f'Predictions: {predictions}')
