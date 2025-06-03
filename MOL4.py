# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# 1. Загрузка и предобработка данных (аналогично 1-й лабораторной)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Удаление ненужных столбцов
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Заполнение пропущенных значений
imputer = SimpleImputer(strategy='median')
data['Age'] = imputer.fit_transform(data[['Age']])
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Преобразование категориальных признаков
label_encoders = {}
for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Разделение на признаки и целевую переменную
X = data.drop('Survived', axis=1)
y = data['Survived']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Случайный лес с оценкой OOB
rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42,
    max_features='sqrt'  # m ≈ sqrt(n_features)
)
rf.fit(X_train, y_train)

# OOB-оценка
oob_error = 1 - rf.oob_score_
print(f"Random Forest OOB Error: {oob_error:.4f}")

# 3. AdaBoost и градиентный бустинг
# AdaBoost
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)

# Градиентный бустинг
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)


# 4. Оценка моделей
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    print(f"\n{name} Performance:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {acc:.4f}, ROC-AUC: {roc_auc:.4f}")

    return fpr, tpr, roc_auc


# Оценка всех моделей
plt.figure(figsize=(10, 8))
for model, name in zip([rf, ada, gb],
                       ['Random Forest', 'AdaBoost', 'Gradient Boosting']):
    fpr, tpr, roc_auc = evaluate_model(model, X_test, y_test, name)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

# Визуализация ROC-кривых
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc="lower right")
plt.savefig('roc_curves.png')
plt.show()


