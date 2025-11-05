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
python3 processing_data_mfa.py

START=$(date +%s)

echo "Step 2: Training VGG16BN model..."
python3 train_model.py

END=$(date +%s)
echo "Step 3: Testing..."
python3 test_model.py

echo "Step 4: Collecting statistics..."
python3 stats.py

echo "Training pipeline completed!"
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
