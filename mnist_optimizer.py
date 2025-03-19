import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from tensorflow.keras.utils import to_categorical

# 1️⃣ MNIST veri setini yükle
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2️⃣ Model fonksiyonu
def create_model(optimizer):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 3️⃣ Optimizer listesi
optimizers = {
    'SGD': SGD(),
    'Adam': Adam(),
    'RMSprop': RMSprop(),
    'Adagrad': Adagrad()
}

results = {}

# 4️⃣ Her optimizer için eğit ve sonuçları kaydet
for name, opt in optimizers.items():
    print(f"\n➡️ {name} ile eğitim başlıyor...")
    model = create_model(opt)
    history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test), verbose=1)
    results[name] = history

# 5️⃣ Sonuçları grafikle göster
plt.figure(figsize=(14, 6))

# Loss
plt.subplot(1, 2, 1)
for name, history in results.items():
    plt.plot(history.history['val_loss'], label=f'{name} val_loss')
plt.title('Validation Loss (MNIST)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
for name, history in results.items():
    plt.plot(history.history['val_accuracy'], label=f'{name} val_acc')
plt.title('Validation Accuracy (MNIST)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
