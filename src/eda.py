import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os

# Настройки визуализации
plt.style.use('ggplot')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)


def load_data():
    """Загрузка всех подготовленных данных"""
    try:
        train_df = pd.read_csv('../data/train_labels.csv')
        val_df = pd.read_csv('../data/val_labels.csv')
        test_df = pd.read_csv('../data/test_labels.csv')

        # Загрузим оригинальные данные для доступа к демографической информации
        original_df = pd.read_csv('../data/Data_Entry_2017.csv')

        return train_df, val_df, test_df, original_df
    except FileNotFoundError as e:
        print(f"Ошибка загрузки файлов: {e}")
        print("Убедитесь, что вы выполнили data_prepare.py сначала")
        return None, None, None, None


def check_patient_overlap(original_df, train_df, val_df, test_df):
    """Проверка пересечения пациентов между наборами данных"""
    print("Проверка пересечения пациентов между наборами:")

    # Получим ID пациентов для каждого набора
    train_patients = set(original_df[original_df['Image Index'].isin(train_df['Image Index'])]['Patient ID'])
    val_patients = set(original_df[original_df['Image Index'].isin(val_df['Image Index'])]['Patient ID'])
    test_patients = set(original_df[original_df['Image Index'].isin(test_df['Image Index'])]['Patient ID'])

    print(f"Train и Val: {len(train_patients.intersection(val_patients))} общих пациентов")
    print(f"Train и Test: {len(train_patients.intersection(test_patients))} общих пациентов")
    print(f"Val и Test: {len(val_patients.intersection(test_patients))} общих пациентов")
    print(
        f"Train, Val и Test: {len(train_patients.intersection(val_patients).intersection(test_patients))} общих пациентов")

    return train_patients, val_patients, test_patients


def analyze_label_distribution(train_df, val_df, test_df, diseases):
    """Анализ распределения меток по наборам данных"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.ravel()

    # Создадим DataFrame для сравнения распределения меток
    comparison_data = []

    datasets = [train_df, val_df, test_df]
    dataset_names = ['Train', 'Validation', 'Test']

    for i, (df, name) in enumerate(zip(datasets, dataset_names)):
        # Подсчет частоты каждого заболевания
        disease_counts = df[diseases].sum().sort_values(ascending=False)

        # Визуализация
        axes[i].barh(disease_counts.index, disease_counts.values)
        axes[i].set_title(f'Распределение заболеваний: {name} ({len(df)} изображений)')
        axes[i].set_xlabel('Количество случаев')

        # Добавим данные для сравнения
        for disease, count in disease_counts.items():
            comparison_data.append({
                'Dataset': name,
                'Disease': disease,
                'Count': count,
                'Percentage': count / len(df) * 100
            })

    # Сравнительная визуализация
    comparison_df = pd.DataFrame(comparison_data)
    pivot_df = comparison_df.pivot(index='Disease', columns='Dataset', values='Percentage')

    # Построим stacked bar chart
    axes[3].barh(pivot_df.index, pivot_df['Train'], label='Train', alpha=0.7)
    axes[3].barh(pivot_df.index, pivot_df['Validation'], left=pivot_df['Train'], label='Validation', alpha=0.7)
    axes[3].barh(pivot_df.index, pivot_df['Test'],
                 left=pivot_df['Train'] + pivot_df['Validation'], label='Test', alpha=0.7)
    axes[3].set_title('Сравнение распределения заболеваний по наборам')
    axes[3].set_xlabel('Процент от общего количества')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig('../outputs/figures/label_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    return comparison_df


def analyze_multi_label_stats(train_df, val_df, test_df, diseases):
    """Анализ multi-label статистик"""
    results = []

    datasets = [train_df, val_df, test_df]
    dataset_names = ['Train', 'Validation', 'Test']

    for df, name in zip(datasets, dataset_names):
        # Количество заболеваний на изображение
        disease_per_image = df[diseases].sum(axis=1)
        avg_diseases = disease_per_image.mean()
        max_diseases = disease_per_image.max()

        # Процент изображений без заболеваний
        no_finding_percent = (df['No Finding'] == 1).mean() * 100

        # Процент изображений с одним заболеванием
        single_disease_percent = (disease_per_image == 1).mean() * 100

        # Процент изображений с множественными заболеваниями
        multi_disease_percent = (disease_per_image > 1).mean() * 100

        results.append({
            'Dataset': name,
            'Total Images': len(df),
            'Avg Diseases per Image': avg_diseases,
            'Max Diseases per Image': max_diseases,
            'No Finding %': no_finding_percent,
            'Single Disease %': single_disease_percent,
            'Multiple Diseases %': multi_disease_percent
        })

        # Визуализация распределения количества заболеваний
        plt.figure(figsize=(10, 6))
        disease_per_image.hist(bins=range(0, int(max_diseases) + 2), alpha=0.7, rwidth=0.8)
        plt.title(f'Распределение количества заболеваний на изображение: {name}')
        plt.xlabel('Количество заболеваний')
        plt.ylabel('Количество изображений')
        plt.savefig(f'../outputs/figures/diseases_per_image_{name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()

    return pd.DataFrame(results)


def analyze_correlations(df, diseases, dataset_name):
    """Анализ корреляций между заболеваниями"""
    # Матрица корреляций
    corr_matrix = df[diseases].corr()

    # Визуализация тепловой карты корреляций
    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title(f'Корреляция между заболеваниями: {dataset_name}')
    plt.savefig(f'../outputs/figures/correlation_{dataset_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Найдем наиболее коррелированные пары заболеваний
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Disease1': corr_matrix.columns[i],
                'Disease2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

    corr_df = pd.DataFrame(corr_pairs)
    return corr_df.sort_values('Correlation', ascending=False)


def analyze_demographics(original_df, train_df, val_df, test_df):
    """Анализ демографических данных"""
    demo_data = []

    datasets = [train_df, val_df, test_df]
    dataset_names = ['Train', 'Validation', 'Test']

    for df, name in zip(datasets, dataset_names):
        # Получим соответствующие демографические данные
        demo_df = original_df[original_df['Image Index'].isin(df['Image Index'])].copy()

        # Анализ распределения по полу
        gender_counts = demo_df['Patient Gender'].value_counts()
        gender_percent = demo_df['Patient Gender'].value_counts(normalize=True) * 100

        # Анализ распределения по возрасту
        # Преобразуем возраст в числовой формат (некоторые значения могут быть строками)
        demo_df['Patient Age'] = pd.to_numeric(demo_df['Patient Age'], errors='coerce')
        avg_age = demo_df['Patient Age'].mean()
        age_std = demo_df['Patient Age'].std()

        demo_data.append({
            'Dataset': name,
            'Male %': gender_percent.get('M', 0),
            'Female %': gender_percent.get('F', 0),
            'Avg Age': avg_age,
            'Age Std': age_std,
            'Total Patients': demo_df['Patient ID'].nunique()
        })

    return pd.DataFrame(demo_data)


def load_data():
    """Загрузка всех подготовленных данных"""
    try:
        train_df = pd.read_csv('../data/train_labels.csv')
        val_df = pd.read_csv('../data/val_labels.csv')
        test_df = pd.read_csv('../data/test_labels.csv')

        # Загрузим оригинальные данные для доступа к демографической информации
        original_df = pd.read_csv('../data/Data_Entry_2017.csv')

        return train_df, val_df, test_df, original_df
    except FileNotFoundError as e:
        print(f"Ошибка загрузки файлов: {e}")
        print("Убедитесь, что вы выполнили data_prepare.py сначала")
        return None, None, None, None

def analyze_patient_distribution(original_df, train_df, val_df, test_df):
    """Анализ распределения снимков по пациентам"""

    # Используем оригинальный датафрейм для получения Patient ID
    # Сопоставляем Image Index с Patient ID из оригинального датафрейма

    def get_patient_stats(original_df, image_indices, dataset_name):
        # Получаем соответствующие данные из оригинального датафрейма
        df_subset = original_df[original_df['Image Index'].isin(image_indices)]

        patient_counts = df_subset['Patient ID'].value_counts()

        stats = {
            'dataset': dataset_name,
            'total_patients': len(patient_counts),
            'total_images': len(df_subset),
            'images_per_patient_avg': patient_counts.mean(),
            'images_per_patient_std': patient_counts.std(),
            'images_per_patient_min': patient_counts.min(),
            'images_per_patient_max': patient_counts.max(),
            'patients_with_1_image': (patient_counts == 1).sum(),
            'patients_with_many_images': (patient_counts > 10).sum()
        }

        return stats, patient_counts

    # Собираем статистику по всем наборам
    train_stats, train_patient_counts = get_patient_stats(
        original_df, train_df['Image Index'].values, 'Train'
    )
    val_stats, val_patient_counts = get_patient_stats(
        original_df, val_df['Image Index'].values, 'Validation'
    )
    test_stats, test_patient_counts = get_patient_stats(
        original_df, test_df['Image Index'].values, 'Test'
    )

    # Визуализация распределения
    plt.figure(figsize=(15, 10))

    # Гистограмма распределения количества снимков на пациента
    plt.subplot(2, 2, 1)
    plt.hist(train_patient_counts, bins=50, alpha=0.7, label='Train')
    plt.hist(val_patient_counts, bins=50, alpha=0.7, label='Validation')
    plt.hist(test_patient_counts, bins=50, alpha=0.7, label='Test')
    plt.xlabel('Количество снимков на пациента')
    plt.ylabel('Количество пациентов')
    plt.title('Распределение снимков по пациентам')
    plt.legend()
    plt.yscale('log')  # Логарифмическая шкала для лучшей визуализации

    # Боксплот распределения
    plt.subplot(2, 2, 2)
    data = [train_patient_counts.values, val_patient_counts.values, test_patient_counts.values]
    plt.boxplot(data, labels=['Train', 'Validation', 'Test'])
    plt.title('Боксплот распределения снимков по пациентам')
    plt.ylabel('Количество снимков на пациента')

    # Топ-20 пациентов с наибольшим количеством снимков
    plt.subplot(2, 2, 3)
    top_patients = train_patient_counts.head(20)
    plt.bar(range(len(top_patients)), top_patients.values)
    plt.xticks(range(len(top_patients)), top_patients.index, rotation=45)
    plt.title('Топ-20 пациентов (Train set)')
    plt.ylabel('Количество снимков')

    # Cumulative distribution
    plt.subplot(2, 2, 4)
    sorted_counts = np.sort(train_patient_counts.values)[::-1]
    cumulative = np.cumsum(sorted_counts) / np.sum(sorted_counts)
    plt.plot(range(1, len(cumulative) + 1), cumulative)
    plt.xlabel('Пациенты (отсортированы по количеству снимков)')
    plt.ylabel('Доля от общего количества снимков')
    plt.title('Кумулятивное распределение снимков по пациентам')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('../outputs/figures/patient_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Сводная статистика
    stats_df = pd.DataFrame([train_stats, val_stats, test_stats])
    print("\n=== РАСПРЕДЕЛЕНИЕ ПО ПАЦИЕНТАМ ===")
    print(stats_df.to_string(index=False))

    return stats_df


def main():
    """Основная функция EDA"""
    print("Загрузка данных...")
    train_df, val_df, test_df, original_df = load_data()

    # Проверим, что данные загрузились успешно
    if train_df is None:
        return

    # Получим список заболеваний (исключим служебные колонки)
    diseases = [col for col in train_df.columns if col not in ['Image Index', 'image_path']]

    # Создадим папку для результатов
    os.makedirs('../outputs/figures', exist_ok=True)

    print("1. Проверка пересечения пациентов между наборами...")
    train_patients, val_patients, test_patients = check_patient_overlap(
        original_df, train_df, val_df, test_df
    )

    print("\n2. Анализ распределения меток...")
    label_stats = analyze_label_distribution(
        train_df, val_df, test_df, diseases
    )

    print("\n3. Анализ multi-label статистик...")
    multi_label_stats = analyze_multi_label_stats(
        train_df, val_df, test_df, diseases
    )

    print("\n4. Анализ корреляций между заболеваниями...")
    # Для корреляционного анализа объединим все данные
    all_df = pd.concat([train_df, val_df, test_df])
    correlation_stats = analyze_correlations(all_df, diseases, 'All')

    print("\n5. Анализ демографических данных...")
    demographic_stats = analyze_demographics(
        original_df,
        train_df, val_df, test_df
    )

    # Сохраним все результаты в CSV
    label_stats.to_csv('../outputs/label_statistics.csv', index=False)
    multi_label_stats.to_csv('../outputs/multi_label_statistics.csv', index=False)
    correlation_stats.to_csv('../outputs/correlation_statistics.csv', index=False)
    demographic_stats.to_csv('../outputs/demographic_statistics.csv', index=False)

    # Выведем сводку результатов
    print("\n" + "=" * 50)
    print("СВОДКА РЕЗУЛЬТАТОВ EDA")
    print("=" * 50)

    print("\nРАСПРЕДЕЛЕНИЕ ДАННЫХ:")
    print(multi_label_stats[['Dataset', 'Total Images', 'Avg Diseases per Image', 'Multiple Diseases %']].to_string(
        index=False))

    print("\nДЕМОГРАФИЧЕСКИЕ ХАРАКТЕРИСТИКИ:")
    print(demographic_stats.to_string(index=False))

    print("\nСАМЫЕ КОРРЕЛИРОВАННЫЕ ПАРЫ ЗАБОЛЕВАНИЙ:")
    print(correlation_stats.head(10).to_string(index=False))

    print("\n6. Анализ распределения снимков по пациентам...")
    patient_stats = analyze_patient_distribution(original_df, train_df, val_df, test_df)

    print(f"\nРезультаты сохранены в папке ../outputs/")
    print("Визуализации сохранены в папке ../outputs/figures/")


if __name__ == "__main__":
    main()