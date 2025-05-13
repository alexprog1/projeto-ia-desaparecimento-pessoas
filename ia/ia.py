import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Simulação de dados com atributos relevantes
np.random.seed(42)
data = pd.DataFrame({
    'idade': np.random.randint(5, 80, 500),
    'ultima_localizacao_lat': np.random.uniform(-23, -22, 500),
    'ultima_localizacao_long': np.random.uniform(-46, -45, 500),
    'tempo_desaparecido': np.random.randint(1, 100, 500),
    'categoria': np.random.choice([0, 1], 500)  # 0: Caso comum, 1: Caso crítico
})

# Divisão dos dados em treino e teste
X = data[['idade', 'ultima_localizacao_lat', 'ultima_localizacao_long', 'tempo_desaparecido']]
y = data['categoria']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Testando diferentes hiperparâmetros no modelo de RandomForest
hyperparameters = [
    {'n_estimators': 10, 'max_depth': 5},
    {'n_estimators': 50, 'max_depth': 10},
    {'n_estimators': 100, 'max_depth': None}
]

for params in hyperparameters:
    model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Hiperparâmetros: {params} -> Acurácia: {acc:.4f}")