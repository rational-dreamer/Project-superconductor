import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Параметры
batch_size = 8
image_size = (128, 128)
initial_learning_rate = 0.001

def load_config(config_file='config.json'):
    with open(config_file) as f:
        return json.load(f)
        
config = load_config()

train_csv = config['learning_dir']+'/train_data.csv'
val_csv = config['learning_dir']+'/val_data.csv'

# Загрузка предварительно обученных весов
encoder_weights_file = 'encoder_weights.npy'
encoder_weights = np.load(encoder_weights_file, allow_pickle=True).item()

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

# Создание tf.data.Dataset
def create_dataset(images, params, batch_size=8, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(({"image": images}, params))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_dataset = create_dataset(train_images, train_params, batch_size=batch_size)
val_dataset = create_dataset(val_images, val_params, batch_size=batch_size, shuffle=False)

def unet_full_model_with_weights(input_image_shape=(3, 128, 128), encoder_weights=None):
    image_input = tf.keras.layers.Input(shape=input_image_shape, name="image")
    c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), data_format="channels_first", name="conv2d_1")(image_input)
    p1 = tf.keras.layers.MaxPooling2D((2, 2), data_format="channels_first")(c1)
    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), data_format="channels_first", name="conv2d_2")(p1)
    p2 = tf.keras.layers.MaxPooling2D((2, 2), data_format="channels_first")(c2)
    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), data_format="channels_first", name="conv2d_3")(p2)
    p3 = tf.keras.layers.MaxPooling2D((2, 2), data_format="channels_first")(c3)
    bottleneck = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), data_format="channels_first", name="bottleneck")(p3)
    flatten = tf.keras.layers.Flatten()(bottleneck)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
    output = tf.keras.layers.Dense(4, activation='linear', name="output")(dense2)

    model = tf.keras.models.Model(inputs=image_input, outputs=output)

    if encoder_weights is not None:
        for layer in model.layers:
            if layer.name in encoder_weights:
                weight_shape_model = [w.shape for w in layer.get_weights()]
                weight_shape_loaded = [w.shape for w in encoder_weights[layer.name]]
                if weight_shape_model == weight_shape_loaded:
                    layer.set_weights(encoder_weights[layer.name])
                else:
                    print(f"Shape mismatch for layer {layer.name}: "f"model expects {weight_shape_model}, weights provided {weight_shape_loaded}")
    
    return model

# Загрузка модели
encoder_weights = np.load("encoder_weights.npy", allow_pickle=True).item()
model = unet_full_model_with_weights(input_image_shape=(3, 128, 128), encoder_weights=encoder_weights)


# Создание модели
model = unet_full_model_with_weights(input_image_shape=(3, 128, 128), encoder_weights=encoder_weights)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), loss='mse', metrics=['mae'])

# Обучение модели
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
    ModelCheckpoint("best_model_with_l2.h5", save_best_only=True)
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

# График функции потерь
plt.plot(history.history['loss'], label='Потери на обучении')
plt.plot(history.history['val_loss'], label='Потери на валидации')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.show()
plt.savefig('loss_graph.png')
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)


# Оценка модели на тестовых данных
print("Оценка модели на тестовых данных...")
test_loss, test_mae = model.evaluate(val_dataset)  # Используем валидационные данные, либо загрузите тестовые
print(f"Validation Loss: {test_loss}, Validation MAE: {test_mae}")

# Предсказание на тестовых данных
print("Предсказание параметров...")
predictions = model.predict(val_dataset)

# Сохранение предсказанных параметров в CSV файл
print("Сохранение результатов предсказаний...")
predicted_df = pd.DataFrame(predictions, columns=["D_pred", "V_pred", "tb_pred", "tp_pred"])
predicted_df["D_true"] = val_params[:,0]
predicted_df["V_true"] = val_params[:,1]
predicted_df["tb_true"] = val_params[:,2]
predicted_df["tp_true"] = val_params[:,3]

# Сохранение в CSV
output_csv = "predicted_params_true_params.csv"
predicted_df.to_csv(output_csv, index=False)

print(f"Предсказанные параметры сохранены в '{output_csv}'")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Построение scatter plots
fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
param_names = ['D', 'V', 'tb', 'tp']
for i, ax in enumerate(axes1.flat):
    ax.scatter(val_params[:, i], predictions[:, i], alpha=0.5)
    ax.plot([val_params[:, i].min(), val_params[:, i].max()],
            [val_params[:, i].min(), val_params[:, i].max()], 'r--')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.set_title(f'{param_names[i]}')
plt.tight_layout()
plt.savefig('predictions_scatter.png')
plt.show()

def calculate_metrics(y_true, y_pred, param_names):
    """
    Расчет метрик качества для каждого параметра
    """
    metrics_dict = {}
    
    for i, param_name in enumerate(param_names):
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        
        metrics_dict[param_name] = {
            'RMSE': np.sqrt(mean_squared_error(true_vals, pred_vals)),
            'MAE': mean_absolute_error(true_vals, pred_vals),
            'R2': r2_score(true_vals, pred_vals),
            'MSE': mean_squared_error(true_vals, pred_vals),
            'Mean_True': np.mean(true_vals),
            'Std_True': np.std(true_vals),
            'Mean_Pred': np.mean(pred_vals),
            'Std_Pred': np.std(pred_vals)
        }
    
    return metrics_dict

def plot_metrics(metrics_dict, param_names):
    """
    Построение графиков метрик
    """
    # Подготовка данных для графиков
    rmse_values = [metrics_dict[param]['RMSE'] for param in param_names]
    r2_values = [metrics_dict[param]['R2'] for param in param_names]
    mae_values = [metrics_dict[param]['MAE'] for param in param_names]
    
    # График RMSE
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    bars = plt.bar(param_names, rmse_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('RMSE по параметрам')
    plt.ylabel('RMSE')
    plt.grid(True, alpha=0.3)
    
    # Добавление значений на столбцы
    for bar, value in zip(bars, rmse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    # График R²
    plt.subplot(1, 3, 2)
    bars = plt.bar(param_names, r2_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('R² Score по параметрам')
    plt.ylabel('R²')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Добавление значений на столбцы
    for bar, value in zip(bars, r2_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # График MAE
    plt.subplot(1, 3, 3)
    bars = plt.bar(param_names, mae_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('MAE по параметрам')
    plt.ylabel('MAE')
    plt.grid(True, alpha=0.3)
    
    # Добавление значений на столбцы
    for bar, value in zip(bars, mae_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_metrics_table(metrics_dict, param_names):
    """
    Создание таблицы с метриками
    """
    table_data = []
    
    for param in param_names:
        metrics = metrics_dict[param]
        table_data.append([
            param,
            f"{metrics['RMSE']:.6f}",
            f"{metrics['MAE']:.6f}",
            f"{metrics['R2']:.4f}",
            f"{metrics['MSE']:.6f}",
            f"{metrics['Mean_True']:.4f} ± {metrics['Std_True']:.4f}",
            f"{metrics['Mean_Pred']:.4f} ± {metrics['Std_Pred']:.4f}"
        ])
    
    # Создание DataFrame
    df = pd.DataFrame(table_data, columns=[
        'Parameter', 'RMSE', 'MAE', 'R²', 'MSE', 
        'True Values (Mean ± Std)', 'Predicted Values (Mean ± Std)'
    ])
    
    # Расчет общих метрик
    overall_rmse = np.sqrt(mean_squared_error(val_params, predictions))
    overall_r2 = r2_score(val_params, predictions)
    overall_mae = mean_absolute_error(val_params, predictions)
    
    # Добавление строки с общими метриками
    overall_row = pd.DataFrame([[
        'OVERALL', 
        f"{overall_rmse:.6f}", 
        f"{overall_mae:.6f}", 
        f"{overall_r2:.4f}", 
        f"{mean_squared_error(val_params, predictions):.6f}",
        '-', '-'
    ]], columns=df.columns)
    
    df = pd.concat([df, overall_row], ignore_index=True)
    
    return df

def print_detailed_analysis(metrics_dict, param_names):
    """
    Подробный анализ результатов
    """
    print("=" * 80)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ КАЧЕСТВА ПРЕДСКАЗАНИЙ")
    print("=" * 80)
    
    for param in param_names:
        metrics = metrics_dict[param]
        print(f"\nПараметр: {param}")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  MAE: {metrics['MAE']:.6f}")
        print(f"  R²: {metrics['R2']:.4f}")
        print(f"  Истинные значения: {metrics['Mean_True']:.4f} ± {metrics['Std_True']:.4f}")
        print(f"  Предсказания: {metrics['Mean_Pred']:.4f} ± {metrics['Std_Pred']:.4f}")
        
        # Анализ качества
        if metrics['R2'] > 0.8:
            quality = "ОТЛИЧНО"
        elif metrics['R2'] > 0.6:
            quality = "ХОРОШО"
        elif metrics['R2'] > 0.4:
            quality = "УДОВЛЕТВОРИТЕЛЬНО"
        else:
            quality = "ПЛОХО"
        
        print(f"  Качество: {quality}")
        
        # Относительные ошибки
        rel_rmse = metrics['RMSE'] / metrics['Mean_True'] * 100 if metrics['Mean_True'] != 0 else float('inf')
        rel_mae = metrics['MAE'] / metrics['Mean_True'] * 100 if metrics['Mean_True'] != 0 else float('inf')
        
        print(f"  Относительная RMSE: {rel_rmse:.2f}%")
        print(f"  Относительная MAE: {rel_mae:.2f}%")

def plot_predictions_vs_true(y_true, y_pred, param_names):
    """
    Графики предсказаний против истинных значений
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, param in enumerate(param_names):
        row, col = i // 2, i % 2
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        
        # Scatter plot
        scatter = axes[row, col].scatter(true_vals, pred_vals, alpha=0.6, c=true_vals, cmap='viridis')
        axes[row, col].plot([true_vals.min(), true_vals.max()], 
                           [true_vals.min(), true_vals.max()], 'r--', lw=2, label='Ideal')
        
        # Линейная регрессия для тренда
        z = np.polyfit(true_vals, pred_vals, 1)
        p = np.poly1d(z)
        axes[row, col].plot(true_vals, p(true_vals), 'b-', alpha=0.8, label=f'Trend (R²={r2_score(true_vals, pred_vals):.3f})')
        
        axes[row, col].set_xlabel(f'True {param}')
        axes[row, col].set_ylabel(f'Predicted {param}')
        axes[row, col].set_title(f'{param}: Predictions vs True Values')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
        
        # Добавление цветовой шкалы
        plt.colorbar(scatter, ax=axes[row, col])
    
    plt.tight_layout()
    plt.savefig('unet_predictions_vs_true.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_distribution(y_true, y_pred, param_names):
    """
    Графики распределения ошибок
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, param in enumerate(param_names):
        row, col = i // 2, i % 2
        errors = y_true[:, i] - y_pred[:, i]
        rel_errors = (errors / (y_true[:, i] + 1e-8)) * 100  # Относительные ошибки в %
        
        # Гистограмма абсолютных ошибок
        axes[row, col].hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[row, col].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        axes[row, col].axvline(x=np.mean(errors), color='g', linestyle='-', linewidth=2, 
                              label=f'Mean: {np.mean(errors):.4f}')
        axes[row, col].set_xlabel(f'Absolute Error ({param})')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].set_title(f'Error Distribution for {param}')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('unet_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
# Основной код выполнения
param_names = ['D', 'V', 'tb', 'tp']

# Расчет метрик
metrics_dict = calculate_metrics(val_params, predictions, param_names)

# Построение графиков
plot_metrics(metrics_dict, param_names)

# Графики предсказаний vs истинных значений
plot_predictions_vs_true(val_params, predictions, param_names)

# Графики распределения ошибок
plot_error_distribution(val_params, predictions, param_names)

# Создание и вывод таблицы
metrics_table = create_metrics_table(metrics_dict, param_names)
print("\nТАБЛИЦА МЕТРИК КАЧЕСТВА:")
print("=" * 100)
print(metrics_table.to_string(index=False))

# Сохранение таблицы в CSV
metrics_table.to_csv('quality_metrics_table.csv', index=False)
print("\nТаблица сохранена в файл 'quality_metrics_table.csv'")

# Подробный анализ
print_detailed_analysis(metrics_dict, param_names)

# Дополнительная визуализация: сравнение RMSE со стандартным отклонением
plt.figure(figsize=(10, 6))
rmse_values = [metrics_dict[param]['RMSE'] for param in param_names]
std_values = [metrics_dict[param]['Std_True'] for param in param_names]

x = np.arange(len(param_names))
width = 0.35

plt.bar(x - width/2, rmse_values, width, label='RMSE', color='lightcoral')
plt.bar(x + width/2, std_values, width, label='Std True Values', color='skyblue')

plt.xlabel('Параметры')
plt.ylabel('Значения')
plt.title('Сравнение RMSE со стандартным отклонением истинных значений')
plt.xticks(x, param_names)
plt.legend()
plt.grid(True, alpha=0.3)

# Добавление значений на столбцы
for i, (rmse, std) in enumerate(zip(rmse_values, std_values)):
    plt.text(i - width/2, rmse + 0.001, f'{rmse:.3f}', ha='center', va='bottom')
    plt.text(i + width/2, std + 0.001, f'{std:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('rmse_vs_std_comparison.png', dpi=300, bbox_inches='tight')
plt.show()