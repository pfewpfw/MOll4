import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
import matplotlib.pyplot as plt

# ==============================
# Этап 1. Загрузка и предварительная обработка данных
# ==============================

# Читаем датасет (предполагается, что это Titanic, как в задании 2)
df = pd.read_csv('D:/для работ/MMOlab/titanic_data.csv')

# Выводим все столбцы
pd.options.display.max_columns = None

# Заполним пропущенные значения:
#   - Для числовых столбцов заполняем медианой,
#   - Для категориальных – модой.
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Масштабируем числовые данные в диапазон [0, 1]
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Преобразуем категориальные переменные в dummy-переменные
df = pd.get_dummies(df, drop_first=True)

# ==============================
# Этап 2. Подготовка данных для классификации
# ==============================

# Для задачи классификации в качестве целевой переменной используем столбец 'Survived'
if 'Survived' not in df.columns:
    raise ValueError("В датасете отсутствует столбец 'Survived'")

# Определяем X (признаки) и y (целевой признак)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Разбиваем данные на обучающую и тестовую выборки (например, 70% на обучение и 30% на тест)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==============================
# Этап 3. Модель случайного леса (Random Forest)
# ==============================

rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)

# Рассчитываем метрики для модели случайного леса
rf_f1 = f1_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)

print("----- Случайный лес -----")
print("F1 Score:    {:.3f}".format(rf_f1))
print("Recall:      {:.3f}".format(rf_recall))
print("Precision:   {:.3f}".format(rf_precision))

# ==============================
# Этап 4. Модель градиентного бустинга (Gradient Boosting)
# ==============================

gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)

# Рассчитываем метрики для модели градиентного бустинга
gb_f1 = f1_score(y_test, gb_pred)
gb_recall = recall_score(y_test, gb_pred)
gb_precision = precision_score(y_test, gb_pred)

print("\n----- Градиентный бустинг -----")
print("F1 Score:    {:.3f}".format(gb_f1))
print("Recall:      {:.3f}".format(gb_recall))
print("Precision:   {:.3f}".format(gb_precision))

# ==============================
# Этап 5. Визуальное сравнение результатов
# ==============================

models = ['Random Forest', 'Gradient Boosting']
f1_scores = [rf_f1, gb_f1]
recalls = [rf_recall, gb_recall]
precisions = [rf_precision, gb_precision]

plt.figure(figsize=(14, 4))

# Сравнение F1-score
plt.subplot(1, 3, 1)
plt.bar(models, f1_scores, color=['skyblue', 'lightgreen'])
plt.title('F1 Score')
plt.ylabel('Score')

# Сравнение Recall
plt.subplot(1, 3, 2)
plt.bar(models, recalls, color=['skyblue', 'lightgreen'])
plt.title('Recall')

# Сравнение Precision
plt.subplot(1, 3, 3)
plt.bar(models, precisions, color=['skyblue', 'lightgreen'])
plt.title('Precision')

plt.suptitle("Сравнение моделей: случайный лес vs. градиентный бустинг", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
