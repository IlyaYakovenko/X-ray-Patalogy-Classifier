import torch.nn as nn
import torch
from torchvision import models
from torchvision.models import ResNet50_Weights


class ChestXRayModel(nn.Module):
    def __init__(self, num_classes=15, pretrained=True, freeze_backbone=True):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        # Загрузка модели
        if self.pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
            print("✓ Загружены предобученные веса ImageNet")
        else:
            self.backbone = models.resnet50(weights=None)
            print("✓ Модель инициализирована случайными весами")

        # Заморозка слоев
        if self.freeze_backbone and self.pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ Все предобученные слои заморожены")

        # Замена последнего слоя
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, self.num_classes)

        # Разморозка последнего слоя
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

        print(f"✓ Заменен последний слой: {num_ftrs} -> {self.num_classes} нейронов")

        self.diseases = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self._print_model_info()

    def forward(self, x):
        return self.backbone(x)

    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n📊 ИНФОРМАЦИЯ О ПАРАМЕТРАХ:")
        print(f"Общее количество параметров: {total_params:,}")
        print(f"Обучаемых параметров: {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)")

    def setup_training(self, learning_rate=0.001, optimizer_type='adam',
                       weight_decay=1e-4):

        # Получаем список заболеваний из датасета
        self.diseases = self.diseases  # Будет установлен извне

        # Выбираем функцию потерь (без весов!)

        criterion = nn.BCEWithLogitsLoss()
        loss_name = "BCEWithLogitsLoss"

        # Оптимизатор
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(trainable_params, lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Неизвестный оптимизатор: {optimizer_type}")

        print(f"\n✓ Настройки обучения:")
        print(f"  - Оптимизатор: {optimizer_type} (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"  - Функция потерь: {loss_name}")

        return optimizer, criterion

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)
        print(f"✓ Модель сохранена: {filepath}")

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=self.device))
        self.to(self.device)
        print(f"✓ Веса модели загружены: {filepath}")
