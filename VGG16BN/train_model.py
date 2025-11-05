import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
from dataset import PreprocessedDataset
from VGG16BN import modified_model
from torch.utils.data import DataLoader

def load_config(config_file='config.json'):
    with open(config_file) as f:
        return json.load(f)
        
config = load_config()

# Пути и гиперпараметры
train_csv = config['learning_dir']+'/train_data.csv'
val_csv = config['learning_dir']+'/val_data.csv'
weights_path = config['learning_dir']+'/model_weights.pth'
metrics_path = config['learning_dir']+'/metrics.png'

batch_size = 16
num_epochs = 200
learning_rate = 1e-4

# Проверка доступного устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Используемое устройство: {device}')

# Инициализация модели
print('Инициализация модели')
model = modified_model()
model.to(device)                                                 # Перенос модели на GPU
criterion = nn.MSELoss()                                         # Функция потерь для регрессии
optimizer = optim.Adam(model.parameters(), lr=learning_rate)     # Оптимизатор


# Данные для обучения и валидации
print('Загрузка данных')
dataset_train = PreprocessedDataset(csv_file=train_csv)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size)

dataset_val = PreprocessedDataset(csv_file=val_csv)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size)

# Цикл обучения
print('Начало обучения')

train_losses, val_losses = [], []

for epoch in range(num_epochs):
    
    # Обучение модели
    model.train()       # Перевод в режим обучения
    train_loss = 0.0
    
    for batch_images, batch_params in dataloader_train:
        
        # Перенос данных на используемое устройство
        batch_images = batch_images.to(device)
        batch_params = batch_params.to(device)

        # Обнуление градиентов
        optimizer.zero_grad()
        
        # Прямой проход
        outputs = model(batch_images)           # Предсказания модели       
        loss = criterion(outputs, batch_params) # Вычисление функции потерь
        
        # Обратный проход 
        loss.backward()
        optimizer.step() # Оптимизация
        
        # Сохраняем значение потерь
        train_loss += loss.item()
    
    train_losses.append(train_loss / len(dataloader_train)) # Средняя потеря за эпоху

    # Валидация
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
         for batch_images, batch_params in dataloader_val:

            # Перенос данных на используемое устройство
            batch_images = batch_images.to(device)
            batch_params = batch_params.to(device)

            # Предсказания
            outputs = model(batch_images)
            loss = criterion(outputs, batch_params)

            # Сохраняем значение потерь
            val_loss += loss.item()

    val_losses.append(val_loss / len(dataloader_val))   # Средняя потеря за эпоху

    # Печать результатов за эпоху
    print(f'Эпоха [{epoch + 1}/{num_epochs}],\
        Потеря при обучении: {train_losses[-1]:.4f},\
        Потеря при валидации: {val_losses[-1]:.4f}')

os.makedirs(os.path.dirname(weights_path), exist_ok=True)
torch.save(model.state_dict(), weights_path)
print(f'Сохранение весов модели: {weights_path}')

# Построение и сохранения графика обучения
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Потери на обучении')
plt.plot(val_losses, label='Потери на валидации')
plt.title('Кривая обучения: Потери на обучении и валидации')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()

os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
plt.savefig(metrics_path)
print(f'График сохранен: {metrics_path}')