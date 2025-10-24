import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit


DISEASES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
    'No Finding'
]


def balance_no_finding(df, max_ratio=0.3, random_state=42):
    no_finding_mask = df['No Finding'] == 1
    pathology_mask = df['No Finding'] == 0

    no_finding_df = df[no_finding_mask]
    pathology_df = df[pathology_mask]

    print(f" До балансировки:")
    print(f"  - С патологиями: {len(pathology_df)} примеров")
    print(f"  - No Finding: {len(no_finding_df)} примеров")
    print(f"  - Доля No Finding: {len(no_finding_df) / len(df):.2%}")

    max_no_finding = int(len(pathology_df) * max_ratio / (1 - max_ratio))

    if len(no_finding_df) > max_no_finding:
        no_finding_df = no_finding_df.sample(n=max_no_finding, random_state=random_state)
        print(f"✓ No Finding ограничено: {len(no_finding_df)} примеров ({max_ratio * 100}%)")
    else:
        print(f"✓ No Finding уже сбалансировано: {len(no_finding_df)} примеров")

    balanced_df = pd.concat([pathology_df, no_finding_df])
    balanced_df = balanced_df.sample(frac=1, random_state=random_state)

    print(f" После балансировки:")
    print(f"  - Всего примеров: {len(balanced_df)}")
    print(f"  - Доля No Finding: {len(no_finding_df) / len(balanced_df):.2%}")

    return balanced_df


def analyze_class_distribution(df, dataset_name):
    print(f"\n РАСПРЕДЕЛЕНИЕ КЛАССОВ ({dataset_name}):")
    total = len(df)

    for disease in DISEASES:
        count = df[disease].sum()
        percentage = (count / total) * 100
        print(f"  {disease:<20}: {count:>5} примеров ({percentage:>5.1f}%)")



def prepare_data():

    df = pd.read_csv('../data/Data_Entry_2017.csv')


    df['image_path'] = df['Image Index'].apply(lambda x: os.path.join('../data', 'images', x))


    for disease in DISEASES:
        df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)


    with open('../data/train_val_list_NIH.txt', 'r') as f:
        train_val_names = [line.strip() for line in f]

    with open('../data/test_list_NIH.txt', 'r') as f:
        test_names = [line.strip() for line in f]


    train_val_df = df[df['Image Index'].isin(train_val_names)]
    test_df = df[df['Image Index'].isin(test_names)]


    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(train_val_df, groups=train_val_df['Patient ID']))


    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]


    balanced_train_df = balance_no_finding(train_df, max_ratio=0.3)


    analyze_class_distribution(balanced_train_df, "TRAIN (сбалансированный)")
    analyze_class_distribution(val_df, "VALIDATION (оригинальный)")
    analyze_class_distribution(test_df, "TEST (оригинальный)")


    columns_to_save = ['Image Index', 'image_path'] + DISEASES

    balanced_train_df[columns_to_save].to_csv('../data/train_labels.csv', index=False)
    val_df[columns_to_save].to_csv('../data/val_labels.csv', index=False)
    test_df[columns_to_save].to_csv('../data/test_labels.csv', index=False)


    train_patients = set(balanced_train_df['Patient ID'])
    val_patients = set(val_df['Patient ID'])
    test_patients = set(test_df['Patient ID'])

    print("\n" + "=" * 60)
    print("ДАННЫЕ ПОДГОТОВЛЕНЫ!")
    print("=" * 60)
    print(f"Train: {len(balanced_train_df)} изображений, {len(train_patients)} пациентов")
    print(f"Validation: {len(val_df)} изображений, {len(val_patients)} пациентов")
    print(f"Test: {len(test_df)} изображений, {len(test_patients)} пациентов")
    print(f"Пересечение пациентов Train/Val: {len(train_patients.intersection(val_patients))}")
    print(f"Пересечение пациентов Train/Test: {len(train_patients.intersection(test_patients))}")
    print(f"Пересечение пациентов Val/Test: {len(val_patients.intersection(test_patients))}")


if __name__ == "__main__":
    prepare_data()
