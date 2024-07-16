import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Özel fonksiyonu tanımlayın ve kayıt edin
@keras.saving.register_keras_serializable()
def mse(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true, y_pred)

# Modeli ve scaler'ı yükleme
autoencoder = load_model('C:/Users/sudec/OneDrive/Masaüstü/Masaüstü Dosyaları/Coding/Python/mnt2/data/autoencoder_model.h5', custom_objects={'mse': mse})
encoder = load_model('C:/Users/sudec/OneDrive/Masaüstü/Masaüstü Dosyaları/Coding/Python/mnt2/data/encoder_model.h5')
scaler = joblib.load('C:/Users/sudec/OneDrive/Masaüstü/Masaüstü Dosyaları/Coding/Python/mnt2/data/scaler.pkl')
kmeans = joblib.load('C:/Users/sudec/OneDrive/Masaüstü/Masaüstü Dosyaları/Coding/Python/mnt2/data/kmeans.pkl')

# Test verisini yükleme
test_data = pd.read_excel('C:/Users/sudec/OneDrive/Masaüstü/Masaüstü Dosyaları/Coding/Python/meyve_oznitellikleri.xlsx')

# 'Renk Ortalaması (BGR)' sütununu 3 ayrı sütuna ayırma
test_data[['renk_B', 'renk_G', 'renk_R']] = test_data['Renk Ortalaması (BGR)'].apply(lambda x: pd.Series(eval(x)))

# Eğitim sırasında kullanılan tüm sütunları dahil etmek için test verisini aynı sütunlarla ayarlayın
required_columns = scaler.feature_names_in_
for col in required_columns:
    if col not in test_data.columns:
        test_data[col] = 0

# Sayısal sütunları seçme (metin sütunları hariç)
numeric_test_data = test_data[required_columns]

# NaN değerleri kontrol etme ve temizleme
if numeric_test_data.isnull().values.any():
    print("NaN değerler tespit edildi ve temizleniyor.")
    numeric_test_data = numeric_test_data.fillna(numeric_test_data.mean())

# Veriyi normalleştirme
test_data_scaled = scaler.transform(numeric_test_data)

# Gizli temsilleri çıkarma
encoded_test_data = encoder.predict(test_data_scaled)

# K-Means kümeleme
test_labels = kmeans.predict(encoded_test_data)

# Test verisine küme etiketlerini eklemek için 'cluster' adında yeni bir sütun oluşturun
test_data['cluster'] = test_labels

# Sonuçları görselleştirme
pca = PCA(n_components=2)
test_data_pca = pca.fit_transform(encoded_test_data)

plt.figure()
scatter = plt.scatter(test_data_pca[:, 0], test_data_pca[:, 1], c=test_labels, cmap='viridis')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('K-Means Kümeleme Sonuçları (Test Verisi)')
plt.colorbar(scatter, ticks=range(kmeans.n_clusters), label='Cluster')
plt.show()

# Sonuçları inceleme
num_clusters = len(np.unique(test_labels))
for cluster in np.unique(test_labels):
    print(f"Küme {cluster}:")
    print(test_data[test_data['cluster'] == cluster].head())

# Silhouette Skoru hesaplama
if num_clusters > 1:
    silhouette_avg = silhouette_score(encoded_test_data, test_labels)
    print(f'Silhouette Skoru (Test Verisi): {silhouette_avg:.2f}')
else:
    print("Silhouette Skoru hesaplanamadı: Etiket sayısı 1")

# Testin başarısını hesaplama
test_loss = autoencoder.evaluate(test_data_scaled, test_data_scaled, verbose=0)
test_success = (1 - test_loss) * 100

# Sonuçları bir DataFrame'e yazma
results_df = pd.DataFrame({
    'Test Loss': [test_loss],
    'Silhouette Score': [silhouette_avg],
    'Test Success (%)': [test_success]
})

# Excel dosyasına yazma
output_file_path = 'try_results2.xlsx'
with pd.ExcelWriter(output_file_path) as writer:
    results_df.to_excel(writer, sheet_name='Test Results', index=False)

print(f"Test sonuçları {output_file_path} dosyasına kaydedildi.")
