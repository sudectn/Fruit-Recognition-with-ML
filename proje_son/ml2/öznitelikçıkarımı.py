import os
import cv2
import pandas as pd
import numpy as np

def calculate_aspect_ratio(image):
    height, width = image.shape[:2]
    return height / width

def calculate_area_perimeter_ratio(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        return area / perimeter if perimeter != 0 else 0
    return 0

def is_porous(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return 1 if len(contours) > 1 else 0

def calculate_color_mean(image):
    mean = cv2.mean(image)[:3]
    return [round(c, 2) for c in mean]

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    file_name = os.path.basename(image_path)
    aspect_ratio = calculate_aspect_ratio(image)
    area_perimeter_ratio = calculate_area_perimeter_ratio(image)
    porous = is_porous(image)
    color_mean = calculate_color_mean(image)
    return {
        'Dosya': file_name,
        'Y/Genişlik Oranı': aspect_ratio,
        'Alan/Çevre Oranı': area_perimeter_ratio,
        'Gözenekli': porous,
        'Renk Ortalaması (BGR)': color_mean
    }

def process_images_in_folder(folder_path, output_file):
    features = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, file_name)
            feature = extract_features(image_path)
            if feature:
                features.append(feature)

    df = pd.DataFrame(features)
    df.to_excel(output_file, index=False)
    print(f"Öznitelikler {output_file} dosyasına kaydedildi.")

# Kullanım
folder_path = 'C:/Users/sudec/OneDrive/Masaüstü/try_meyve'  # Klasör yolunu güncelleyin
output_file = 'meyve_oznitellikleri.xlsx'
process_images_in_folder(folder_path, output_file)
