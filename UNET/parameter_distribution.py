import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

image_size = (128, 128)

def load_config(config_file='config.json'):
    with open(config_file) as f:
        return json.load(f)
        
config = load_config()

train_csv = config['learning_dir']+'/train_data.csv'
val_csv = config['learning_dir']+'/val_data.csv'


# Функция для загрузки изображений
def load_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    return img_to_array(img, data_format="channels_first") / 255.0

# Функция для загрузки данных из CSV
def load_data_from_csv(csv_path, target_size=(128, 128)):
    data = pd.read_csv(csv_path)
    images = []
    params = []
    image_paths = []
    for _, row in data.iterrows():
        img_path = row['Path']
        img = load_image(img_path, target_size=target_size)
        images.append(img)
        params.append([row['D'], row['V'], row['tb'], row['tp']])
        image_paths.append(img_path)
    return np.array(images), np.array(params), image_paths

# Загрузка данных
train_images, train_params, train_paths = load_data_from_csv(train_csv, target_size=image_size)
val_images, val_params, val_paths = load_data_from_csv(val_csv, target_size=image_size)

param_names = ['Δ', 'V', 'tb', 'tp']

plt.figure(figsize=(12, 8))

for i, param_name in enumerate(param_names):
    plt.subplot(2, 2, i+1)
    
    # Данные для текущего параметра
    param_data = train_params[:, i]
    
    # Построение гистограммы
    n, bins, patches = plt.hist(param_data, bins=30, alpha=0.7, edgecolor='black')
    
    # Добавление вертикальной линии для среднего значения
    mean_val = np.mean(param_data)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_val:.3f}')
    
    # Добавление вертикальной линии для медианы
    median_val = np.median(param_data)
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                label=f'Median: {median_val:.3f}')
    
    plt.xlabel(f'Value of {param_name}')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of parameter {param_name}')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('parameter_distribution_histograms.png', dpi=300, bbox_inches='tight')
plt.show()