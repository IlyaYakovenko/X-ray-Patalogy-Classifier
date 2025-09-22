from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np


class ChestXRayDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        """
        Конструктор класса ChestXRayDataset

        Args:
            csv_path (str): Путь к CSV-файлу с метками и путями к изображениям
            transform (callable, optional): Функция/трансформация для применения к изображениям
        """
        # Загружаем CSV-файл с метками и путями к изображениям
        self.df = pd.read_csv(csv_path)

        # Сохраняем трансформации для применения к изображениям
        self.transform = transform

        # Получаем список заболеваний из колонок CSV (исключаем 'Image Index' и 'image_path')
        self.diseases = [col for col in self.df.columns if col not in ['Image Index', 'image_path']]

    def __len__(self):
        """
        Возвращает количество элементов в датасете
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Возвращает один элемент данных по индексу

        Args:
            idx (int): Индекс элемента

        Returns:
            tuple: (image, labels) - изображение и соответствующие метки
        """
        # Получаем строку данных по индексу
        row = self.df.iloc[idx]

        # Извлекаем путь к изображению из строки
        image_path = row['image_path']

        # Загружаем изображение с помощью OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Преобразуем изображение из оттенков серого в RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Извлекаем метки для всех заболеваний
        labels = np.array([row[disease] for disease in self.diseases], dtype=np.float32)

        # Применяем трансформации к изображению, если они указаны
        if self.transform:
            image = self.transform(image)

        # Возвращаем изображение и соответствующие метки
        return image, labels

    def get_diseases(self):
        """
        Вспомогательный метод для получения списка заболеваний

        Returns:
            list: Список названий заболеваний
        """
        return self.diseases