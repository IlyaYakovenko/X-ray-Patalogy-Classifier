import os
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data import BalancedAugmentationChestXRayDataset, ChestXRayDataset
from src.model import ChestXRayModel


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._create_directories()

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None

        self.history = {
            'train_loss': [], 'val_loss': [], 'val_auc': [],
            'val_macro_f1': [], 'val_micro_f1': [], 'learning_rates': []
        }

        print("ИНИЦИАЛИЗАЦИЯ ТРЕНЕРА")
        print(f"Конфигурация: {json.dumps(config, indent=2)}")

    def _create_directories(self):
        os.makedirs('/content/drive/MyDrive/X-ray/outputs/models', exist_ok=True)
        os.makedirs('/content/drive/MyDrive/X-ray/outputs/figures', exist_ok=True)
        os.makedirs('/content/drive/MyDrive/X-ray/outputs/logs', exist_ok=True)

    def setup_data(self):
        print("\nПОДГОТОВКА ДАННЫХ")

        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        val_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        train_dataset = BalancedAugmentationChestXRayDataset(
            csv_path='/content/drive/MyDrive/X-ray/data/train_labels.csv',
            transform=train_transform
        )

        val_dataset = ChestXRayDataset(
            csv_path='/content/drive/MyDrive/X-ray/data/val_labels.csv',
            transform=val_transform
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        self.diseases = train_dataset.diseases
        print(f"Тренировочный набор: {len(train_dataset)} изображений")
        print(f"Валидационный набор: {len(val_dataset)} изображений")

    def setup_model(self):
        print("\nНАСТРОЙКА МОДЕЛИ")

        self.model = ChestXRayModel(
            num_classes=len(self.diseases),
            pretrained=self.config['pretrained'],
            freeze_backbone=self.config['freeze_backbone']
        )

        self.model.diseases = self.diseases

        self.optimizer, self.criterion = self.model.setup_training(
            learning_rate=self.config['learning_rate'],
            optimizer_type=self.config['optimizer'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Батч {batch_idx}/{len(self.train_loader)}, Потеря: {loss.item():.4f}")

        return running_loss / num_batches

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                preds = torch.sigmoid(outputs)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                running_loss += loss.item()

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        val_loss = running_loss / len(self.val_loader)
        val_auc = self.calculate_auc(all_labels, all_preds)
        threshold = self.config.get('threshold', 0.3)
        macro_f1, micro_f1 = self.calculate_f1(all_labels, all_preds, threshold)

        print("\nAUC по классам:")
        auc_scores = []
        for i, disease in enumerate(self.diseases):
            try:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                auc_scores.append(auc)

                if auc > 0.7:
                    print(f"{disease}: {auc:.4f}")
                elif auc < 0.6:
                    print(f"{disease}: {auc:.4f}")
                else:
                    print(f"   {disease}: {auc:.4f}")
            except ValueError:
                auc_scores.append(0.5)
                print(f"{disease}: невозможно вычислить AUC")

        return val_loss, val_auc, macro_f1, micro_f1

    def calculate_auc(self, labels, preds):
        auc_scores = []
        for i in range(labels.shape[1]):
            if len(np.unique(labels[:, i])) >= 2:
                try:
                    auc = roc_auc_score(labels[:, i], preds[:, i])
                    auc_scores.append(auc)
                except ValueError:
                    auc_scores.append(0.5)
        return np.mean(auc_scores) if auc_scores else 0.5

    def calculate_f1(self, labels, preds, threshold=0.3):
        from sklearn.metrics import f1_score, classification_report
        binary_preds = (preds > threshold).astype(int)
        macro_f1 = f1_score(labels, binary_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(labels, binary_preds, average='micro', zero_division=0)

        if labels.shape[1] <= 15:
            print(classification_report(labels, binary_preds, target_names=self.diseases, zero_division=0))

        return macro_f1, micro_f1

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config,
            'diseases': self.diseases
        }

        checkpoint_path = f"/content/drive/MyDrive/X-ray/outputs/models/checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_model_path = "/content/drive/MyDrive/X-ray/outputs/models/best_model.pth"
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Лучшая модель сохранена: {best_model_path}")

    def plot_training_history(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.history['val_auc'], label='Val AUC', color='green')
        ax2.set_title('Validation AUC')
        ax2.legend()
        ax2.grid(True)

        ax3.plot(self.history['val_macro_f1'], label='Val Macro F1', color='orange')
        ax3.plot(self.history['val_micro_f1'], label='Val Micro F1', color='red')
        ax3.set_title('Validation F1-Scores')
        ax3.legend()
        ax3.grid(True)

        ax4.plot(self.history['learning_rates'], label='Learning Rate', color='red')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/X-ray/outputs/figures/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Графики сохранены")

    def train(self):
        print("\nЗАПУСК ОБУЧЕНИЯ")
        print("=" * 60)

        self.setup_data()
        self.setup_model()

        best_auc = 0.0
        start_time = time.time()

        for epoch in range(1, self.config['num_epochs'] + 1):
            epoch_start = time.time()

            print(f"\nЭПОХА {epoch}/{self.config['num_epochs']}")
            print("-" * 50)

            train_loss = self.train_epoch()
            val_loss, val_auc, macro_f1, micro_f1 = self.validate_epoch()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_auc)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            self.history['val_macro_f1'].append(macro_f1)
            self.history['val_micro_f1'].append(micro_f1)
            self.history['learning_rates'].append(current_lr)

            print(f"\nРЕЗУЛЬТАТЫ ЭПОХИ {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val AUC:    {val_auc:.4f}")
            print(f"  Val F1:     Macro: {macro_f1:.4f}, Micro: {micro_f1:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Время эпохи: {time.time() - epoch_start:.2f} сек")

            if epoch % 5 == 0:
                self.save_checkpoint(epoch)

            if val_auc > best_auc:
                best_auc = val_auc
                self.save_checkpoint(epoch, is_best=True)
                print(f" AUC: {best_auc:.4f}")

        total_time = time.time() - start_time
        print(f"\nОБУЧЕНИЕ ЗАВЕРШЕНО за {total_time / 60:.1f} мин")
        print(f"Лучшая AUC: {best_auc:.4f}")

        self.plot_training_history()
        self.save_checkpoint(self.config['num_epochs'])

        history_df = pd.DataFrame(self.history)
        history_df.to_csv('/content/drive/MyDrive/X-ray/outputs/logs/training_history.csv', index=False)

        with open('/content/drive/MyDrive/X-ray/outputs/logs/training_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)

        print("Все результаты сохранены")


TRAINING_CONFIG = {
    'num_epochs': 10,
    'batch_size': 128,
    'learning_rate': 0.00982119621793305,
    'image_size': 224,
    'pretrained': True,
    'freeze_backbone': True,
    'optimizer': 'adam',
    'num_workers': 2,
    'threshold': 0.1,
    'weight_decay': 1.7744079336785624e-05,
}


trainer = Trainer(TRAINING_CONFIG)

try:
    trainer.train()
except Exception as e:
    print(f"\nОшибка во время обучения: {e}")