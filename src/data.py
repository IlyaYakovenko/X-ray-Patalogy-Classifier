from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ChestXRayDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.diseases = [col for col in self.df.columns if col not in ['Image Index', 'image_path']]

    def __len__(self):
        return len(self.df)

    def _load_and_preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']

        image = self._load_and_preprocess_image(image_path)
        labels = np.array([row[disease] for disease in self.diseases], dtype=np.float32)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, labels

    def get_diseases(self):
        return self.diseases


class BalancedAugmentationChestXRayDataset(ChestXRayDataset):
    def __init__(self, csv_path, transform=None):
        super().__init__()
        self.original_dataset = ChestXRayDataset(csv_path, transform=None)
        self.transform = transform
        self.diseases = self.original_dataset.diseases

        self.class_distribution = self._analyze_class_distribution()
        self.max_class_count = max(self.class_distribution.values())

        self.augmentation_plan = self._create_augmentation_plan()
        self.augmented_samples = self._prepare_augmented_samples()

        for disease in self.diseases:
            original_count = self.class_distribution[disease]
            augmented_count = len([s for s in self.augmented_samples if s['disease'] == disease])
            print(f"  {disease}: {original_count} → {original_count + augmented_count}")

    def _analyze_class_distribution(self):
        class_counts = {}
        df = self.original_dataset.df

        for disease in self.diseases:
            class_counts[disease] = int(df[disease].sum())

        return class_counts

    def _create_augmentation_plan(self):
        augmentation_plan = {}

        for disease, count in self.class_distribution.items():
            needed_augmentations = self.max_class_count - count
            augmentation_plan[disease] = max(0, needed_augmentations)

        return augmentation_plan

    def _prepare_augmented_samples(self):
        augmented_samples = []
        df = self.original_dataset.df

        for disease, num_augmentations in self.augmentation_plan.items():
            if num_augmentations == 0:
                continue

            class_indices = df[df[disease] == 1].index.tolist()

            if not class_indices:
                continue

            augmentations_per_sample = num_augmentations // len(class_indices)
            remainder = num_augmentations % len(class_indices)

            for i, idx in enumerate(class_indices):
                num_for_this_sample = augmentations_per_sample + (1 if i < remainder else 0)

                for aug_id in range(num_for_this_sample):
                    augmented_samples.append({
                        'original_index': idx,
                        'disease': disease,
                        'augmentation_id': aug_id
                    })

        return augmented_samples

    def __len__(self):
        return len(self.original_dataset) + len(self.augmented_samples)

    def _apply_augmentation(self, image, augmentation_id):
        base_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.8),
            A.RandomBrightnessContrast(p=0.3),
        ])

        if augmentation_id % 3 == 0:
            transform = A.Compose([
                base_transform,
                A.Affine(translate_percent=0.05, scale=(0.9, 1.1), p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        elif augmentation_id % 3 == 1:
            transform = A.Compose([
                base_transform,
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            transform = A.Compose([
                base_transform,
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

        return transform(image=image)['image']

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            image, labels = self.original_dataset[idx]
            is_augmented = False
        else:
            aug_info = self.augmented_samples[idx - len(self.original_dataset)]
            image, labels = self.original_dataset[aug_info['original_index']]
            image = self._apply_augmentation(image, aug_info['augmentation_id'])
            is_augmented = True

        if not is_augmented and self.transform:
            image = self.transform(image=image)['image']

        return image, labels