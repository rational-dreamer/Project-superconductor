import os
import torch
import pandas as pd
import json
from sklearn.metrics import mean_squared_error
from dataset import PreprocessedDataset
from torch.utils.data import DataLoader
from VGG16BN import modified_model

# Пути и гиперпараметры
def load_config(config_file='config.json'):
    with open(config_file) as f:
        return json.load(f)
        
config = load_config()

# Пути и гиперпараметры
weights_path = config['learning_dir']+'/model_weights.pth'
test_csv = config['learning_dir']+'/test_data.csv'
predictions_csv = config['learning_dir']+'/predictions.csv'

batch_size = 16

# Инициализация модели
print('Загрузка модели')
model = modified_model()
model.load_state_dict(torch.load(weights_path, weights_only=True))
model.eval()

# Данные для тестирования
print('Загрузка данных')
dataset_test = PreprocessedDataset(csv_file=test_csv)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size)


# Инициализация метрик
criterion = torch.nn.MSELoss()

# Тестирование модели
print('Начало тестирования')
all_predictions = []
all_labels = []
total_loss = 0.0

with torch.no_grad():
    for batch_images, batch_params in dataloader_test:

        # Передаем изображения через модель
        outputs = model(batch_images)

        # Рассчитываем потери
        loss = criterion(outputs, batch_params)
        total_loss += loss.item()

        # Сохраняем предсказания и настоящие значения для последующего анализа
        all_predictions.append(outputs)
        all_labels.append(batch_params)

# Вычисляем среднее значение потерь
average_loss = total_loss / len(dataset_test)
print(f"Средняя потеря на тестовом наборе: {average_loss:.4f}")

# Преобразуем списки в тензоры для дальнейшего анализа
all_predictions = torch.cat(all_predictions, dim=0)
all_labels = torch.cat(all_labels, dim=0)

# Вычисляем MSE
mse = mean_squared_error(all_labels.numpy(), all_predictions.numpy())
print(f"Среднеквадратичная ошибка (MSE): {mse:.4f}")

# Сохранение предсказаний
print(f'Сохранение предсказаний в файл: {predictions_csv}')
df = pd.DataFrame(
    all_predictions, columns=['Dpr', 'Vpr', 'tbpr', 'tppr']
)

os.makedirs(os.path.dirname(predictions_csv), exist_ok=True)
df.to_csv(predictions_csv, index=False)
