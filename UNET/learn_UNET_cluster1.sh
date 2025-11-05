#!/bin/bash

# Проверяем, передан ли аргумент с конфигурационным файлом
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config.json>"
    exit 1
fi

CONFIG_FILE=$1

# Проверяем существование файла
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file $CONFIG_FILE not found!"
    exit 1
fi

# Копируем конфиг в текущую директорию, если он еще не там
if [ "$CONFIG_FILE" != "config.json" ]; then
    cp "$CONFIG_FILE" config.json
fi

# Выполняем шаги обучения

echo "Step 1: Processing data..."
python3 processing_data_cluster1.py

echo "Step 2: Training UNET_FULL model..."
python3 UNET_FULL_2.py

echo "Step 3: Training UNET_WE model..."
python3 UNET_WE_5.py

echo "Step 4: Creating dataset..."
python3 dataset_creation.py

echo "Training pipeline completed!"
