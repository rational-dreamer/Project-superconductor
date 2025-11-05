import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Параметры
batch_size = 8
image_size = (128, 128)  # Размер изображений
initial_learning_rate = 0.01  # Увеличено для ускорения переобучения

def load_config(config_file='config.json'):
    with open(config_file) as f:
        return json.load(f)
        
config = load_config()
learning_dir = config['learning_dir']
train_csv = learning_dir+'/train_data.csv'
val_csv = learning_dir+'/val_data.csv'
test_csv = learning_dir+'/test_data.csv'
        
# Функция для загрузки изображений
def load_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    return img_to_array(img, data_format="channels_first") / 255.0

# Функция для загрузки данных из CSV
def load_data_from_csv(csv_path, target_size=(128, 128)):
    data = pd.read_csv(csv_path)
    images = []
    params = []
    image_paths = []  # Список путей изображений
    
    for _, row in data.iterrows():
        img_path = row['Path']
        img = load_image(img_path, target_size=target_size)
        images.append(img)
        params.append([row['D'], row['V'], row['tb'], row['tp']])
        image_paths.append(img_path)
    
    return np.array(images), np.array(params), image_paths

# Загрузка данных
print("Загрузка тренировочных данных...")
train_images, train_params, train_paths = load_data_from_csv(train_csv, target_size=image_size)
print("Загрузка валидационных данных...")
val_images, val_params, val_paths = load_data_from_csv(val_csv, target_size=image_size)
print("Загрузка тестовых данных...")
test_images, test_params, test_paths = load_data_from_csv(test_csv, target_size=image_size)

# Создание tf.data.Dataset
def create_dataset(images, params, batch_size=8, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(({"image": images}, params))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_images, train_params, batch_size=batch_size, shuffle=True)
val_dataset = create_dataset(val_images, val_params, batch_size=batch_size, shuffle=False)
test_dataset = create_dataset(test_images, test_params, batch_size=batch_size, shuffle=False)

# Модель для переобучения
def unet_full_model(input_image_shape=(3, 128, 128)):
    image_input = tf.keras.layers.Input(shape=input_image_shape, name="image")

    # Encoder
    c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_first")(image_input)
    p1 = tf.keras.layers.MaxPooling2D((2, 2), data_format="channels_first")(c1)

    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first")(p1)
    p2 = tf.keras.layers.MaxPooling2D((2, 2), data_format="channels_first")(c2)

    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first")(p2)
    p3 = tf.keras.layers.MaxPooling2D((2, 2), data_format="channels_first")(c3)

    c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_first")(p3)
    p4 = tf.keras.layers.MaxPooling2D((2, 2), data_format="channels_first")(c4)

    bottleneck = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first")(p4)

    # Decoder
    d1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format="channels_first")(bottleneck)
    d2 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format="channels_first")(d1)
    d3 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format="channels_first")(d2)
    d4 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format="channels_first")(d3)

    gap = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_first")(d4)
    dense1 = tf.keras.layers.Dense(256, activation='relu')(gap)
    dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
    output = tf.keras.layers.Dense(4, activation='linear', name="output")(dense2)

    model = tf.keras.models.Model(inputs=image_input, outputs=output)
    return model

# Создание модели
full_model = unet_full_model(input_image_shape=(3, 128, 128))

# Компиляция
full_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), loss='mse', metrics=['mae'])

# Обучение
history = full_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    verbose=1,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
)

# Сохранение весов энкодера
encoder_weights = {layer.name: layer.get_weights() for layer in full_model.layers if "conv2d" in layer.name}
np.save('encoder_weights.npy', encoder_weights)

# Построение графиков
train_losses = history.history['loss']
val_losses = history.history['val_loss']

#plt.figure(figsize=(10, 5))
#plt.plot(train_losses, label='Train Loss')
#plt.plot(val_losses, label='Validation Loss')
#plt.legend()
#plt.show()
