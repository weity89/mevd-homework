import pandas as pd
import re
from typing import Tuple

from pip._internal import distributions


# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    fsuir_df = data[data['факультет'] == 'факультет систем управления и робототехники']
    students_count = len(fsuir_df)
    groups_count = fsuir_df['группа'].nunique()
    return students_count, groups_count, fsuir_df


# Задача 2
def find_homonymous_students(df: pd.DataFrame) -> Tuple[bool, int, pd.Series, str]:
    fsuir_df = df.copy()
    fsuir_df['фамилия'] = fsuir_df['фио'].str.split().str[0]
    fsuir_df = fsuir_df.dropna(subset=['фамилия'])

    surname_counts = fsuir_df['фамилия'].value_counts()
    homonyms = surname_counts[surname_counts > 1]

    has_homonyms = len(homonyms) > 0
    if not has_homonyms:
        return False, 0, pd.Series(dtype=int), ""

    total_homonyms = int(homonyms.sum())

    homonym_students = fsuir_df[fsuir_df['фамилия'].isin(homonyms.index)]

    distribution_by_course = {}
    for course, group in homonym_students.groupby('курс'):
        course_surname_counts = group['фамилия'].value_counts()
        course_homonyms = course_surname_counts[course_surname_counts > 1]
        distribution_by_course[course] = course_homonyms.sum()

    distribution_by_course = pd.Series(distribution_by_course).sort_index()

    group_counts = homonym_students.groupby('группа').size()
    max_group = group_counts.idxmax() if not group_counts.empty else ""

    return has_homonyms, total_homonyms, distribution_by_course, max_group

# Задача 3
def gender_identification(patronym: str) -> str:
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """
    if not isinstance(patronym, str) or pd.isna(patronym) or patronym.strip() == "":
        return "unknown"

    p = patronym.strip().lower()

    if p.endswith(('овна', 'евна', 'ична', 'инична')):
        return "female"
    elif p.endswith(('ович', 'евич', 'ич')):
        return "male"

    return "unknown"


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    def get_gender_and_patronym(fio):
        if not isinstance(fio, str):
            return "unknown", False

        parts = fio.strip().split()

        if len(parts) < 3:
            return "unknown", False

        patronym = parts[2]
        gender = gender_identification(patronym)

        return gender, True

    results = df['фио'].apply(get_gender_and_patronym)

    genders = results.apply(lambda x: x[0])
    has_patronym = results.apply(lambda x: x[1])

    no_patronym_count = int((~has_patronym).sum())

    gender_distribution = genders[genders != 'unknown'].value_counts()

    return no_patronym_count, gender_distribution


# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    counts = data['факультет'].value_counts()

    max_faculty = (counts.index[0], counts.iloc[0])
    min_faculty = (counts.index[-1], counts.iloc[-1])

    return counts, max_faculty, min_faculty


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    counts = data.groupby(['факультет', 'курс']).size()

    mean_students = counts.groupby(level='курс').mean()
    median_students = counts.groupby(level='курс').median()

    return mean_students, median_students


# Задача 6
def most_popular_name(data: pd.DataFrame) -> Tuple[str, str, str, int, float]:
    """
    Определяет самое популярное имя, группу с наибольшим количеством студентов с этим именем,
    факультет, курс и долю таких студентов в общем числе.
    Возвращает результат в следующем порядке:
     1. самое частое имя
     2. группа
     3. факультет
     4. доля
    """
    data = data.copy()
    data['имя'] = data['фио'].str.split().str[1]

    name_counts = data['имя'].value_counts()
    most_popular = name_counts.index[0]
    total_with_name = name_counts.iloc[0]

    students_with_name = data[data['имя'] == most_popular]

    group_counts = students_with_name.groupby('группа').size()
    best_group_name = students_with_name.groupby('группа').size().idxmax()

    best_group_info = students_with_name[students_with_name['группа'] == best_group_name].iloc[0]
    faculty = best_group_info['факультет']
    course = best_group_info['курс']

    total_students = len(data)
    share = round(total_with_name / total_students, 2)

    return most_popular, best_group_name, faculty, course, share


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    df = data.copy()

    df['имя'] = df['фио'].str.split().str[1]

    name_counts = df['имя'].value_counts()

    unique_names = name_counts[name_counts == 1].index

    mask = df['имя'].isin(unique_names) & df['имя'].str.startswith('П', na=False)

    result = df[mask]

    return result[['фио', 'факультет', 'курс']]


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл котого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    df_3 = data[data['курс'] == '3-й'].copy()

    faculty_avg = df_3.groupby('факультет')['средний_балл'].mean()
    best_faculty = faculty_avg.idxmax()

    best_faculty_df = df_3[df_3['факультет'] == best_faculty].copy()

    def get_gender(fio):
        if not isinstance(fio, str):
            return "unknown"
        parts = fio.strip().split()
        if len(parts) >= 3:
            return gender_identification(parts[2])
        return "unknown"

    best_faculty_df['gender'] = best_faculty_df['фио'].apply(get_gender)

    gender_avg = best_faculty_df.groupby('gender')['средний_балл'].mean()

    gender_avg = gender_avg[gender_avg.index.isin(['male', 'female'])]
    best_gender = gender_avg.idxmax()
    best_score = int(round(gender_avg.max()))

    return best_faculty, best_gender, best_score


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    df_sorted = data.sort_values('ису').copy()
    df_sorted['diff'] = df_sorted['ису'].diff()

    diff = df_sorted['diff'].fillna(0)
    mask = (diff == 1)

    for i in range(len(df_sorted) - 4):
        if mask.iloc[i + 1:i + 5].all():
            result = df_sorted.iloc[i:i + 5]
            return result[['фио', 'факультет', 'курс', 'группа', 'ису']].reset_index(drop=True)

    return pd.DataFrame(columns=['фио', 'факультет', 'курс', 'группа', 'ису'])


if __name__ == "__main__":
    data = pd.read_csv("isu_fake_data.csv")
    # data["surname"] = "put your code here"
    # data["name"] = "put your code here"
    # data["patronim"] = "put your code here"

    # Задача 1
    num_students, num_groups, fsuir = filter_fsuir_students(data)
    print(f"Студентов на ФСУиР: {num_students}, Групп: {num_groups}")

    # Задача 2
    has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group = find_homonymous_students(fsuir)
    print(f"Есть однофамильцы: {has_homonyms}, Всего: {total_homonyms}, Группа с максимумом: {max_homonym_group}")
    print(f"На каждом курсе: {homonyms_per_course}")

    # Задача 3
    students_without_patronym, gender_counts = analyze_patronyms(fsuir)
    print(f"Студентов без отчества: {students_without_patronym}")
    print("Распределение по полу:", gender_counts)

    # Задача 4
    faculty_counts, max_faculty, min_faculty = faculty_statistics(data)
    print(f"Факультет с наибольшим числом студентов: {max_faculty}")
    print(f"Факультет с наименьшим числом студентов: {min_faculty}")

    # Задача 5
    mean_students, median_students = course_statistics(data)
    print("Среднее число студентов на курсах:", mean_students)
    print("Медианное число студентов на курсах:", median_students)

    # Задача 6
    popular_name, name_group, faculty, course, name_ratio = most_popular_name(data)
    print(f"Самое популярное имя: {popular_name}, Группа: {name_group}, Факультет: {faculty}, Курс: {course}")
    print(f"Доля студентов с этим именем: {name_ratio}")

    # Задача 7
    result_7 = find_students_with_name_starting_P(data)
    print("Студенты с именем, начинающимся на П и встречающимся ровно один раз:")
    print(result_7)

    # Задача 8
    fac, best_gender, best_grade = highest_avg_grade_faculty(data)
    print(f"Факультет с высоким средним баллом 3-го курса: {fac}")
    print(f"Пол с наивысшим средним баллом: {best_gender}, Средний балл: {best_grade}")

    # Задача 9
    result_9 = find_consecutive_students(data)
    print("Первые 5 студентов с подряд идущими табельными номерами:")
    print(result_9)
