
# Importa o NumPy para trabalhar com vetores
import numpy as np

# Importa as métricas de classificação
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)


# Função responsável por avaliar as previsões da rede
def avaliar_classificacao(modelo, X_val, y_val, threshold=0.5):

    # Solicita que a rede produza probabilidades de aprovação
    probabilidades = modelo.predict(
        X_val,
        verbose=0
    )

    # Converte as probabilidades em classificações
    # Valores maiores ou iguais ao threshold recebem 1
    # Valores menores que o threshold recebem 0
    previsoes = (probabilidades >= threshold).astype(int)

    # Converte as previsões para um vetor de uma dimensão
    previsoes = previsoes.flatten()

    # Calcula a matriz de confusão
    # A primeira linha representa os reprovados reais
    # A segunda linha representa os aprovados reais
    matriz = confusion_matrix(
        y_val,
        previsoes,
        labels=[0, 1]
    )

    # Calcula a acurácia geral
    acuracia = accuracy_score(
        y_val,
        previsoes
    )

    # Calcula o Recall da classe reprovado (0)
    recall_reprovados = recall_score(
        y_val,
        previsoes,
        pos_label=0,
        zero_division=0
    )

    # Calcula a Precision da classe reprovado (0)
    precisao_reprovados = precision_score(
        y_val,
        previsoes,
        pos_label=0,
        zero_division=0
    )

    # Calcula o F1-score da classe reprovado (0)
    f1_reprovados = f1_score(
        y_val,
        previsoes,
        pos_label=0,
        zero_division=0
    )

    # Mostra o threshold utilizado
    print(f"\n--- THRESHOLD {threshold} ---")

    # Exibe a matriz de confusão
    print("\nMatriz de confusão:")
    print(matriz)

    # Mostra a quantidade prevista para cada classe
    print("Previstos como reprovados:", np.sum(previsoes == 0))
    print("Previstos como aprovados:", np.sum(previsoes == 1))

    # Mostra todas as métricas em porcentagem
    print("\n--- MÉTRICAS ---")
    print(f"Acurácia geral: {acuracia * 100:.2f}%")
    print(f"Recall dos reprovados: {recall_reprovados * 100:.2f}%")
    print(f"Precision dos reprovados: {precisao_reprovados * 100:.2f}%")
    print(f"F1-score dos reprovados: {f1_reprovados * 100:.2f}%")
