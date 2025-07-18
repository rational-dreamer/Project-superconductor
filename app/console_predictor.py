import requests
import argparse
import os

def send_image_to_api(image_path):
    try:
        # Проверяем существование файла
        if not os.path.exists(image_path):
            print(f"Ошибка: файл '{image_path}' не найден")
            return

        # Отправляем изображение на сервер
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f)}
            response = requests.post(
                'http://127.0.0.1:8080/calculate-parameters',
                files=files
            )

        # Обрабатываем ответ
        if response.status_code == 200:
            result = response.json()
            params = result['params']
            print("\nРезультаты предсказания:")
            print(f"D: {params['D']:.4f}")
            print(f"V: {params['V']:.4f}")
            print(f"td: {params['td']:.4f}")
            print(f"tp: {params['tp']:.4f}")
        else:
            print(f"Ошибка сервера: {response.status_code}")
            print(response.json())

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Отправка изображения на API')
    parser.add_argument('image_path', type=str, help='Путь к изображению')
    args = parser.parse_args()
    
    send_image_to_api(args.image_path)