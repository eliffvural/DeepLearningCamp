# Gerekli kütüphaneleri içe aktaralım
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

#MNIST veri setini yükleyelim
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Veriyi normalize edelim (0-255 arasından 0-1 arasına)
x_train, x_test = x_train / 255.0, x_test / 255.0

#Modelimizi oluşturalım (Basit Sequential model)
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),            # 28x28 görselleri tek boyuta indirir
    layers.Dense(128, activation='relu'),            # 128 nöronlu gizli katman
    layers.Dropout(0.2),                             # Aşırı öğrenmeyi engellemek için dropout
    layers.Dense(10, activation='softmax')           # Çıkış katmanı (10 sınıf için softmax)
])

#Modeli derleyelim (optimizer ve loss belirleyelim)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Eğitime başlayalım
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

#Test verisi üzerinde doğruluk
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest doğruluğu: {test_acc*100:.2f}%")

#Eğitim sürecini grafikle gösterelim
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()
