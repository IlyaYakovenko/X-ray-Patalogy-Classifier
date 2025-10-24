import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def check_image_quality(csv_path, sample_size=100000):
    df = pd.read_csv(csv_path)

    stats = {
        'width': [],
        'height': [],
        'mean_intensity': [],
        'std_intensity': [],
        'min_intensity': [],
        'max_intensity': [],
        'corrupted_count': 0
    }


    print(f"Проверка качества {len(df)} изображений")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = row['image_path']

        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                stats['corrupted_count'] += 1
                continue

            stats['width'].append(image.shape[1])
            stats['height'].append(image.shape[0])
            stats['mean_intensity'].append(np.mean(image))
            stats['std_intensity'].append(np.std(image))
            stats['min_intensity'].append(np.min(image))
            stats['max_intensity'].append(np.max(image))

        except Exception as e:
            stats['corrupted_count'] += 1
            print(f"Ошибка при обработке {image_path}: {e}")

    print("\n=== РЕЗУЛЬТАТЫ ПРОВЕРКИ КАЧЕСТВА ===")
    print(f"Битых изображений: {stats['corrupted_count']} ({stats['corrupted_count'] / len(df) * 100:.2f}%)")

    if len(stats['width']) > 0:
        print(f"\nРазмеры изображений:")
        print(f"  Ширина: {np.mean(stats['width']):.1f} ± {np.std(stats['width']):.1f} пикселей")
        print(f"  Высота: {np.mean(stats['height']):.1f} ± {np.std(stats['height']):.1f} пикселей")

        print(f"\nИнтенсивность пикселей:")
        print(f"  Средняя: {np.mean(stats['mean_intensity']):.1f} ± {np.std(stats['mean_intensity']):.1f}")
        print(f"  Стандартное отклонение: {np.mean(stats['std_intensity']):.1f} ± {np.std(stats['std_intensity']):.1f}")
        print(f"  Минимальная: {np.mean(stats['min_intensity']):.1f}")
        print(f"  Максимальная: {np.mean(stats['max_intensity']):.1f}")

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(stats['width'], bins=30, alpha=0.7, color='blue')
        plt.title('Распределение ширины изображений')
        plt.xlabel('Ширина (пиксели)')
        plt.ylabel('Количество')

        plt.subplot(1, 2, 2)
        plt.hist(stats['height'], bins=30, alpha=0.7, color='green')
        plt.title('Распределение высоты изображений')
        plt.xlabel('Высота (пиксели)')
        plt.ylabel('Количество')

        plt.tight_layout()
        plt.savefig('../outputs/figures/image_size_distribution.png', dpi=300, bbox_inches='tight')

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(stats['mean_intensity'], bins=30, alpha=0.7, color='red')
        plt.title('Распределение средней интенсивности')
        plt.xlabel('Интенсивность')
        plt.ylabel('Количество')

        plt.subplot(1, 2, 2)
        plt.hist(stats['std_intensity'], bins=30, alpha=0.7, color='purple')
        plt.title('Распределение стандартного отклонения интенсивности')
        plt.xlabel('Стандартное отклонение')
        plt.ylabel('Количество')

        plt.tight_layout()
        plt.savefig('../outputs/figures/image_intensity_distribution.png', dpi=300, bbox_inches='tight')

        print(f"\nВизуализации сохранены в папке ../outputs/figures/")

    return stats




if __name__ == "__main__":
    os.makedirs('../outputs/figures', exist_ok=True)

    check_image_quality('../data/train_labels.csv')

