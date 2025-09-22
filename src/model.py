import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


def create_resnet50_model(num_classes=15, pretrained=True, freeze_weights=False):
    """
    Создает модель ResNet-50 для multi-label классификации

    Args:
        num_classes: количество классов (заболеваний)
        pretrained: использовать ли предобученные веса ImageNet
        freeze_weights: заморозить ли веса предобученных слоев
    """
    # Загружаем предобученную модель ResNet-50
    if pretrained:
        # Используем самые последние веса (лучшие практики)
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    # Заменяем последний полносвязный слой для нашего количества классов
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Если нужно заморозить веса
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False
        # Размораживаем последний слой для обучения
        for param in model.fc.parameters():
            param.requires_grad = True

    return model


def create_efficientnet_model(num_classes=15, pretrained=True, freeze_weights=False):
    """
    Создает модель EfficientNet-B0 для multi-label классификации
    """
    try:
        # Пытаемся загрузить EfficientNet
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)
        else:
            model = models.efficientnet_b0(weights=None)

        # Заменяем классификатор
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

        # Если нужно заморозить веса
        if freeze_weights:
            for param in model.parameters():
                param.requires_grad = False
            # Размораживаем последний слой для обучения
            for param in model.classifier.parameters():
                param.requires_grad = True

        return model
    except AttributeError:
        print("EfficientNet недоступен в вашей версии torchvision. Используйте torchvision 0.11+")
        return None


# Дополнительные утилиты для работы с моделью
def count_trainable_parameters(model):
    """Подсчитывает количество обучаемых параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_model_optimizer(model, learning_rate=0.001, optimizer_type='adam'):
    """Настраивает оптимизатор для модели"""
    # Собираем параметры, которые требуют градиентов
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(trainable_params, lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Неизвестный оптимизатор: {optimizer_type}")

    return optimizer