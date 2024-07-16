import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import os
import tensorflow.keras as keras
from tensorflow.keras.regularizers import l2

# Veriyi yükleme
data = pd.read_excel('C:/Users/sudec/OneDrive/Masaüstü/Masaüstü Dosyaları/Coding/Python/ml/train_data.xlsx')

# Sayısal sütunları seçme (metin sütunları hariç)
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Veriyi normalleştirme
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(numeric_data)

# Sonuç değişkenini tanımlama
result_column = 'Meyve'  # Sonuç değişkeni olan sütunun adı
result = data[result_column]

# Özel fonksiyonu tanımlayın
def mse(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true, y_pred)

# Autoencoder modeli
input_dim = data_scaled.shape[1]
encoding_dim = 2  # Gizli katman boyutu

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=l2(1e-5))(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

# Autoencoder'ı derleme
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Early stopping callback'i ekleme
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Modeli eğitme
history = autoencoder.fit(data_scaled, data_scaled, epochs=50, batch_size=16, shuffle=True, validation_split=0.2, callbacks=[early_stopping])

# Eğitim ve doğrulama kaybı değerlerini çıktı alma
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Eğitim sonuçlarını bir DataFrame'e dönüştürme
results_df = pd.DataFrame({
    'Epoch': range(1, len(train_loss) + 1),
    'Training Loss': train_loss,
    'Validation Loss': val_loss
})

# Son epoch'taki kayıp değerlerini hesaplama
final_train_loss = train_loss[-1]
final_val_loss = val_loss[-1]
learning_success = (1 - final_val_loss) * 100

# Modelin son performansını DataFrame'e ekleme
summary_df = pd.DataFrame({
    'Final Training Loss': [final_train_loss],
    'Final Validation Loss': [final_val_loss],
    'Learning Success (%)': [learning_success]
})

# Excel dosyasına yazma
output_file_path = 'training_results2.xlsx'
with pd.ExcelWriter(output_file_path) as writer:
    results_df.to_excel(writer, sheet_name='Training History', index=False)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print(f"Eğitim sonuçları {output_file_path} dosyasına kaydedildi.")

print("Eğitim Kaybı Değerleri:")
print(train_loss)
print("\nDoğrulama Kaybı Değerleri:")
print(val_loss)

# Öğrenme eğrisini çizme
plt.figure()
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Öğrenme Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')

# Aşırı öğrenme olup olmadığını kontrol etme
# Aşırı öğrenmenin başladığı epoch'u bulmak için
loss = history.history['loss']
val_loss = history.history['val_loss']

overfitting_epoch = None
for i in range(1, len(loss)):
    if val_loss[i] > val_loss[i-1] and val_loss[i] > loss[i]:
        overfitting_epoch = i
        break

if overfitting_epoch is not None:
    plt.axvline(overfitting_epoch, color='r', linestyle='--', label='Aşırı Öğrenme Başlangıcı')

plt.legend()
plt.show()

# Gizli temsilleri çıkarma
encoded_data = encoder.predict(data_scaled)

# K-Means kümeleme
kmeans = KMeans(n_clusters=8)
kmeans.fit(encoded_data)
labels = kmeans.labels_

# Orijinal veriye küme etiketlerini ekleyin
data['cluster'] = labels

# Her kümeden örnekler seçip inceleyin
for cluster in np.unique(labels):
    print(f"Küme {cluster}:")
    print(data[data['cluster'] == cluster].head())

# Sonuçları görselleştirme
pca = PCA(n_components=2)
data_pca = pca.fit_transform(encoded_data)

plt.figure()
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('K-Means Kümeleme Sonuçları')
plt.show()

# Modelin doğruluğu ve hata oranları
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

print(f"Son Eğitim Kaybı: {train_loss}")
print(f"Son Doğrulama Kaybı: {val_loss}")

# Silhouette Skoru
silhouette_avg = silhouette_score(encoded_data, labels)
print(f'Silhouette Skoru: {silhouette_avg:.2f}')

# Yüzdelik öğrenme başarısı
learning_success = (1 - val_loss) * 100
print(f"Öğrenme Başarısı: %{learning_success:.2f}")

# Modelleri ve scaler'ı kaydetme
autoencoder.save('C:/Users/sudec/OneDrive/Masaüstü/Masaüstü Dosyaları/Coding/Python/mnt2/data/autoencoder_model.h5')
encoder.save('C:/Users/sudec/OneDrive/Masaüstü/Masaüstü Dosyaları/Coding/Python/mnt2/data/encoder_model.h5')
joblib.dump(scaler, 'C:/Users/sudec/OneDrive/Masaüstü/Masaüstü Dosyaları/Coding/Python/mnt2/data/scaler.pkl')
joblib.dump(kmeans, 'C:/Users/sudec/OneDrive/Masaüstü/Masaüstü Dosyaları/Coding/Python/mnt2/data/kmeans.pkl')

print("Eğitilmiş modeller ve scaler başarıyla kaydedildi.")

# Modeli yükleme
loaded_autoencoder = keras.models.load_model('C:/Users/sudec/OneDrive/Masaüstü/Masaüstü Dosyaları/Coding/Python/mnt2/data/autoencoder_model.h5', custom_objects={'mse': mse})

# Küme etiketlerini Excel dosyasına yazma
output_file_path_with_clusters = 'train_data_with_clusters.xlsx'
data.to_excel(output_file_path_with_clusters, index=False)

print(f"Kümeleme sonuçları {output_file_path_with_clusters} dosyasına kaydedildi.")
