import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit

# Список всех заболеваний в датасете
DISEASES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
    'No Finding'
]


# Загрузка и подготовка данных
def prepare_data():
    # Загрузка основного файла с метками
    df = pd.read_csv('../data/Data_Entry_2017.csv')

    # Создаем полный путь к изображениям
    df['image_path'] = df['Image Index'].apply(lambda x: os.path.join('../data', 'images', x))

    # Создаем multi-hot encoding для каждого заболевания
    for disease in DISEASES:
        df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)

    # Загружаем списки разделения на train/val и test
    with open('../data/train_val_list_NIH.txt', 'r') as f:
        train_val_names = [line.strip() for line in f]

    with open('../data/test_list_NIH.txt', 'r') as f:
        test_names = [line.strip() for line in f]

        # Создаем DataFrame для train_val и test
        train_val_df = df[df['Image Index'].isin(train_val_names)]
        test_df = df[df['Image Index'].isin(test_names)]

        # Разделяем train_val на train и validation с учетом пациентов
        # Используем GroupShuffleSplit чтобы гарантировать, что все снимки одного пациента
        # попадут в один набор (train или validation)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        # Получаем индексы для train и validation
        # groups - это массив с ID пациентов для каждого изображения
        train_idx, val_idx = next(gss.split(train_val_df, groups=train_val_df['Patient ID']))

        # Создаем DataFrame для train и validation
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

        # Сохраняем только необходимые колонки
        columns_to_save = ['Image Index', 'image_path'] + DISEASES

        train_df[columns_to_save].to_csv('../data/train_labels.csv', index=False)
        val_df[columns_to_save].to_csv('../data/val_labels.csv', index=False)
        test_df[columns_to_save].to_csv('../data/test_labels.csv', index=False)

        # Проверяем, что пациенты не пересекаются между наборами
        train_patients = set(train_df['Patient ID'])
        val_patients = set(val_df['Patient ID'])
        test_patients = set(test_df['Patient ID'])

        print("Данные подготовлены!")
        print(f"Train: {len(train_df)} изображений, {len(train_patients)} пациентов")
        print(f"Validation: {len(val_df)} изображений, {len(val_patients)} пациентов")
        print(f"Test: {len(test_df)} изображений, {len(test_patients)} пациентов")
        print(f"Пересечение пациентов Train/Val: {len(train_patients.intersection(val_patients))}")
        print(f"Пересечение пациентов Train/Test: {len(train_patients.intersection(test_patients))}")
        print(f"Пересечение пациентов Val/Test: {len(val_patients.intersection(test_patients))}")

if __name__ == "__main__":
    prepare_data()
