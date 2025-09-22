import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd
import numpy as np
from model import create_resnet50_model, setup_model_optimizer, count_trainable_parameters
from data import ChestXRayDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import os
import time

# Гиперпараметры
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 0.001
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Аугментации и преобразования
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


def train_model():
    """Основная функция обучения модели"""
    print(f"Используется устройство: {DEVICE}")

    # Загрузка данных
    train_dataset = ChestXRayDataset('../data/train_labels.csv', transform=train_transform)
    val_dataset = ChestXRayDataset('../data/val_labels.csv', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Создание модели
    model = create_resnet50_model(num_classes=len(train_dataset.diseases), pretrained=True, freeze_weights=False)
    model = model.to(DEVICE)

    # Печатаем информацию о модели
    trainable_params = count_trainable_parameters(model)
    print(f"Модель создана. Обучаемых параметров: {trainable_params:,}")

    # Настройка оптимизатора и функции потерь
    optimizer = setup_model_optimizer(model, learning_rate=LR, optimizer_type='adam')
    criterion = nn.BCEWithLogitsLoss()

    # Для отслеживания лучшей модели
    best_auc = 0
    best_model_path = '../outputs/models/best_model.pth'
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # История обучения
    history = {
        'train_loss': [],
        'val_auc': [],
        'val_f1': []
    }

    print("Начинаем обучение...")
    for epoch in range(NUM_EPOCHS):
        # Фаза обучения
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Обнуляем градиенты
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Обратный проход и оптимизация
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # Фаза валидации
        model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                preds = torch.sigmoid(outputs)  # Преобразуем в вероятности

                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

        # Объединяем результаты
        all_labels = np.vstack(all_labels)
        all_preds = np.vstack(all_preds)

        # Вычисляем метрики
        epoch_loss = running_loss / len(train_loader.dataset)

        # ROC-AUC для каждого класса
        auc_scores = []
        for i in range(all_labels.shape[1]):
            try:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                auc_scores.append(auc)
            except ValueError:
                auc_scores.append(0.0)

        mean_auc = np.mean(auc_scores)

        # F1-score (требуется бинаризация)
        threshold = 0.5
        binarized_preds = (all_preds > threshold).astype(int)
        f1_scores = []
        for i in range(all_labels.shape[1]):
            f1 = f1_score(all_labels[:, i], binarized_preds[:, i], zero_division=0)
            f1_scores.append(f1)

        mean_f1 = np.mean(f1_scores)

        # Сохраняем историю
        history['train_loss'].append(epoch_loss)
        history['val_auc'].append(mean_auc)
        history['val_f1'].append(mean_f1)

        # Сохраняем лучшую модель
        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"Новая лучшая модель сохранена с AUC: {best_auc:.4f}")

        # Выводим статистику
        epoch_time = time.time() - start_time
        print(
            f'Epoch {epoch + 1}/{NUM_EPOCHS}, Time: {epoch_time:.2f}s, Loss: {epoch_loss:.4f}, Val AUC: {mean_auc:.4f}, Val F1: {mean_f1:.4f}')

    # Сохраняем историю обучения
    history_df = pd.DataFrame(history)
    history_df.to_csv('../outputs/training_history.csv', index=False)

    # Сохраняем финальную модель
    final_model_path = '../outputs/models/final_model.pth'
    torch.save(model.state_dict(), final_model_path)

    # Визуализация обучения
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(history['val_auc'], label='AUC')
    plt.plot(history['val_f1'], label='F1')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('../outputs/figures/training_metrics.png', dpi=300, bbox_inches='tight')

    print(f"Обучение завершено. Лучшая AUC: {best_auc:.4f}")
    print(f"Модели сохранены в папке ../outputs/models/")


if __name__ == "__main__":
    train_model()