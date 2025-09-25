import optuna
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from model import ChestXRayModel
from data import BalancedAugmentationChestXRayDataset
from weightedFocalLoss import WeightedFocalLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OptunaTuner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        os.makedirs('/content/drive/MyDrive/X-ray/outputs/optuna', exist_ok=True)

    def objective(self, trial):
        hyperparams = {
            'max_weight': trial.suggest_float('max_weight', 2.0, 8.0, step=0.5),
            'focal_alpha': trial.suggest_float('focal_alpha', 0.4, 0.8, step=0.1),
            'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.0, step=0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'threshold': trial.suggest_float('threshold', 0.1, 0.4, step=0.05),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
            'augmentation_multiplier': trial.suggest_int('augmentation_multiplier', 1, 5),
        }

        print(f"\n🎯 Trial {trial.number}: "
              f"max_weight={hyperparams['max_weight']}, "
              f"alpha={hyperparams['focal_alpha']}, "
              f"gamma={hyperparams['focal_gamma']}, "
              f"lr={hyperparams['learning_rate']:.5f}")

        try:
            best_auc = self.fast_train_evaluate(hyperparams, trial)


            trial.set_user_attr("hyperparams", hyperparams)

            return best_auc

        except Exception as e:
            print(f"❌ Trial {trial.number} failed: {e}")
            return 0.5

    def fast_train_evaluate(self, hyperparams, trial):
        fast_config = {
            'num_epochs': 3,
            'batch_size': hyperparams['batch_size'],
            'learning_rate': hyperparams['learning_rate'],
            'image_size': 224,
            'pretrained': True,
            'freeze_backbone': True,
            'optimizer': 'adam',
            'num_workers': 2,
            'max_weight': hyperparams['max_weight'],
            'focal_alpha': hyperparams['focal_alpha'],
            'focal_gamma': hyperparams['focal_gamma'],
            'threshold': hyperparams['threshold'],
            'weight_decay': hyperparams['weight_decay'],
            'augmentation_multiplier': hyperparams['augmentation_multiplier'],
        }

        train_loader, val_loader = self.create_dataloaders(fast_config)

        model = ChestXRayModel(
            num_classes=15,
            pretrained=fast_config['pretrained'],
            freeze_backbone=fast_config['freeze_backbone']
        )

        optimizer, criterion = self.custom_setup_training(
            model,
            fast_config,
            '/content/drive/MyDrive/X-ray/data/train_labels.csv'
        )

        best_auc = 0.0

        for epoch in range(fast_config['num_epochs']):
            train_loss = self.fast_train_epoch(model, train_loader, optimizer, criterion)

            current_auc = self.fast_validate(model, val_loader, fast_config['threshold'])

            trial.report(current_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if current_auc > best_auc:
                best_auc = current_auc

        del model, optimizer, criterion
        torch.cuda.empty_cache()

        return best_auc

    def custom_setup_training(self, model, config, train_csv_path):
        df_sample = pd.read_csv(train_csv_path)
        diseases = [col for col in df_sample.columns if col not in ['Image Index', 'image_path']]

        df = pd.read_csv(train_csv_path)
        class_counts = df[diseases].sum(axis=0).values
        total_samples = len(df)

        class_weights = total_samples / (len(diseases) * np.maximum(class_counts, 1))
        class_weights = np.minimum(class_weights, config['max_weight'])
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        criterion = WeightedFocalLoss(
            class_weights=class_weights,
            alpha=config['focal_alpha'],
            gamma=config['focal_gamma'],
            reduction='mean'
        )

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        return optimizer, criterion

    def create_dataloaders(self, config):
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
            csv_path='../data/train_labels.csv',
            transform=train_transform,
            augmentation_multiplier=config['augmentation_multiplier']
        )

        val_dataset = BalancedAugmentationChestXRayDataset(
            csv_path='../data/val_labels.csv',
            transform=val_transform,
            augmentation_multiplier=1
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )

        return train_loader, val_loader

    def fast_train_epoch(self, model, train_loader, optimizer, criterion):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx > 30:
                break

            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        return running_loss / num_batches if num_batches > 0 else 0.0

    def fast_validate(self, model, val_loader, threshold):
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                preds = torch.sigmoid(outputs)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        auc_scores = []
        for i in range(all_labels.shape[1]):
            if len(np.unique(all_labels[:, i])) >= 2:
                try:
                    auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                    auc_scores.append(auc)
                except:
                    auc_scores.append(0.5)

        return np.mean(auc_scores) if auc_scores else 0.5

    def run_study(self, n_trials=50):
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1, max_resource=3, reduction_factor=3
            )
        )

        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        self.save_results(study)

        return study

    def save_results(self, study):
        best_params = study.best_trial.params
        best_value = study.best_value

        print(f"\n ЛУЧШИЕ ГИПЕРПАРАМЕТРЫ (AUC: {best_value:.4f}):")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        results = {
            'best_auc': best_value,
            'best_params': best_params,
            'trials_count': len(study.trials)
        }

        with open('../outputs/optuna/best_hyperparams.json', 'w') as f:
            json.dump(results, f, indent=2)

        trials_df = study.trials_dataframe()
        trials_df.to_csv('/content/drive/MyDrive/X-ray/outputs/optuna/all_trials.csv', index=False)

        try:
            import plotly
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_image('/content/drive/MyDrive/X-ray/outputs/optuna/optimization_history.png')

            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image('/content/drive/MyDrive/X-ray/outputs/optuna/param_importances.png')

            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image('/content/drive/MyDrive/X-ray/outputs/optuna/parallel_coordinate.png')
        except Exception as e:
            print(f"  Не удалось создать визуализации: {e}")

        print(" Результаты сохранены в /content/drive/MyDrive/X-ray/outputs/optuna/")

def main():

    base_config = {
        'num_epochs': 10,
        'image_size': 224,
        'pretrained': True,
        'freeze_backbone': True,
        'optimizer': 'adam',
        'num_workers': 2,
    }

    tuner = OptunaTuner(base_config)
    study = tuner.run_study(n_trials=30)

    best_params = study.best_trial.params
    final_config = base_config.copy()
    final_config.update(best_params)
    final_config['num_epochs'] = 10

    with open('../outputs/optuna/final_training_config.json', 'w') as f:
        json.dump(final_config, f, indent=2)

    print(f"\n Подбор завершен! Лучший F1: {study.best_value:.4f}")


if __name__ == "__main__":
    main()
