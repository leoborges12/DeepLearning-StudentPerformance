
# Importa o NumPy para criar as previsões
import numpy as np

# Importa as métricas utilizadas na avaliação
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)


# Função responsável por avaliar o baseline
def avaliar_baseline(y_val):

    # Cria um vetor preenchido com 1
    # Isso significa prever aprovação para todos os alunos
    previsoes = np.ones(
        len(y_val),
        dtype=int
    )

    # Calcula a matriz de confusão
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

    # Calcula o Recall dos reprovados
    recall = recall_score(
        y_val,
        previsoes,
        pos_label=0,
        zero_division=0
    )

    # Calcula a Precision dos reprovados
    precisao = precision_score(
        y_val,
        previsoes,
        pos_label=0,
        zero_division=0
    )

    # Calcula o F1-score dos reprovados
    f1 = f1_score(
        y_val,
        previsoes,
        pos_label=0,
        zero_division=0
    )

    # Exibe os resultados
    print("\n--- BASELINE: SEMPRE PREVER APROVADO ---")

    # Mostra a matriz de confusão
    print("Matriz de confusão:")
    print(matriz)

    # Mostra as métricas em porcentagem
    print(f"Acurácia: {acuracia * 100:.2f}%")
    print(f"Recall dos reprovados: {recall * 100:.2f}%")
    print(f"Precision dos reprovados: {precisao * 100:.2f}%")
    print(f"F1-score dos reprovados: {f1 * 100:.2f}%")
