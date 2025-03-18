import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Örnek veri seti: (X = input, Y = output)
# X → [sıcaklık, nem, rüzgar]
# Y → (piknik yapabilir miyiz? 1: evet, 0: hayır)
X = np.array([
    [30, 85, 10],
    [25, 80, 5],
    [35, 60, 20],
    [20, 90, 0],
    [40, 70, 15]
])

Y = np.array([
    [0],
    [1],
    [1],
    [0],
    [1]
])

#Modeli oluşturuyoruz (3 giriş, 2 gizli katman, 1 çıkış)
model = Sequential()
model.add(Dense(10, input_dim=3, activation='relu'))  # 1. katman
model.add(Dense(5, activation='relu'))               # 2. katman
model.add(Dense(1, activation='sigmoid'))            # Çıkış katmanı (0 veya 1 için)

#Modeli derle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Eğit
model.fit(X, Y, epochs=200, verbose=1)

#Tahmin yap
predict = model.predict(np.array([[28, 75, 7]]))
print(f"Piknik yapma olasılığı: {predict[0][0]:.2f}")
