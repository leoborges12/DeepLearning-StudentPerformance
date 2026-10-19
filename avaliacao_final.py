
# Importa o NumPy para cálculos matemáticos
import numpy as np

# Importa o TensorFlow para executar a rede neural
import tensorflow as tf

# Importa Path para criar e organizar diretórios
from pathlib import Path

# Importa o modelo de Regressão Logística
from sklearn.linear_model import LogisticRegression

# Importa as métricas de classificação
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)

# Importa a função para calcular os pesos das classes
from sklearn.utils.class_weight import compute_class_weight

# Importa o Early Stopping
from tensorflow.keras.callbacks import EarlyStopping

# Importa nossa função de preparação dos dados
from preparacao_dados import preparar_dados

# Importa nossa função de criação da MLP
from modelo import criar_modelo


# -------------------------------------------------
# 1. FUNÇÃO DE AVALIAÇÃO
# -------------------------------------------------

# Define uma função para calcular e apresentar as métricas
def avaliar(nome, y_real, probabilidades, threshold=0.6):

    # Converte as probabilidades em classes 0 ou 1
    previsoes = (probabilidades >= threshold).astype(int)

    # Calcula a matriz de confusão
    matriz = confusion_matrix(
        y_real,
        previsoes,
        labels=[0, 1]
    )

    # Calcula a acurácia geral
    acuracia = accuracy_score(y_real, previsoes)

    # Calcula o recall da classe reprovado (0)
    recall = recall_score(
        y_real,
        previsoes,
        pos_label=0,
        zero_division=0
    )

    # Calcula a precisão da classe reprovado (0)
    precisao = precision_score(
        y_real,
        previsoes,
        pos_label=0,
        zero_division=0
    )

    # Calcula o F1-score da classe reprovado (0)
    f1 = f1_score(
        y_real,
        previsoes,
        pos_label=0,
        zero_division=0
    )

    # Exibe o nome do modelo avaliado
    print(f"\n========== {nome} ==========")

    # Exibe a matriz de confusão
    print("Matriz de confusão:")
    print(matriz)

    # Exibe a acurácia em porcentagem
    print(f"Acurácia: {acuracia * 100:.2f}%")

    # Exibe o recall dos reprovados
    print(f"Recall dos reprovados: {recall * 100:.2f}%")

    # Exibe a precisão dos reprovados
    print(f"Precision dos reprovados: {precisao * 100:.2f}%")

    # Exibe o F1-score dos reprovados
    print(f"F1-score dos reprovados: {f1 * 100:.2f}%")


# -------------------------------------------------
# 2. PREPARAÇÃO DOS DADOS
# -------------------------------------------------

# Prepara os conjuntos de treinamento, validação e teste
X_treino, y_treino, X_val, y_val, X_teste, y_teste = preparar_dados()


# -------------------------------------------------
# 3. REGRESSÃO LOGÍSTICA
# -------------------------------------------------

# Cria o classificador de Regressão Logística
regressao = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# Treina utilizando apenas o conjunto de treinamento
regressao.fit(X_treino, y_treino)

# Localiza a posição da classe aprovado (1)
indice_aprovado = list(regressao.classes_).index(1)

# Calcula as probabilidades de aprovação no teste
prob_regressao = regressao.predict_proba(
    X_teste
)[:, indice_aprovado]


# -------------------------------------------------
# 4. REDE NEURAL MLP 16-8
# -------------------------------------------------

# Limpa os estados anteriores do TensorFlow
tf.keras.backend.clear_session()

# Define a semente aleatória do experimento
tf.keras.utils.set_random_seed(42)

# Cria a rede com 16 e 8 neurônios nas camadas ocultas
modelo = criar_modelo(
    X_treino.shape[1],
    16,
    8
)

# Identifica as classes existentes no treinamento
classes = np.unique(y_treino)

# Calcula os pesos balanceados das classes
pesos = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_treino
)

# Organiza os pesos das classes em um dicionário
pesos_classes = {
    int(classe): float(peso)
    for classe, peso in zip(classes, pesos)
}

# Configura a parada antecipada do treinamento
parada = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# Treina a rede neural utilizando treinamento e validação
historico = modelo.fit(
    X_treino,
    y_treino,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=pesos_classes,
    callbacks=[parada],
    verbose=0
)

# Identifica a época que apresentou a menor perda de validação
melhor_epoca = int(np.argmin(historico.history["val_loss"]) + 1)

# Exibe informações sobre o treinamento
print("\n========== TREINAMENTO DA MLP ==========")
print("Épocas executadas:", len(historico.history["loss"]))
print("Melhor época:", melhor_epoca)
print("Parâmetros treináveis:", modelo.count_params())


# -------------------------------------------------
# 5. SALVAMENTO DA REDE NEURAL
# -------------------------------------------------

# Define a pasta onde vamos salvar a rede neural
pasta_modelos = Path("modelos")

# Cria a pasta caso ainda não exista
pasta_modelos.mkdir(exist_ok=True)

# Define o caminho completo do arquivo da rede
caminho_modelo = pasta_modelos / "mlp_student_performance.keras"

# Salva a arquitetura, os pesos e a configuração da MLP
modelo.save(caminho_modelo)

# Confirma que a rede neural foi salva
print("\n========== MODELO SALVO ==========")
print("Arquivo:", caminho_modelo)


# -------------------------------------------------
# 6. PREVISÕES NO CONJUNTO DE TESTE
# -------------------------------------------------

# Calcula as probabilidades de aprovação dos alunos de teste
prob_mlp = modelo.predict(
    X_teste,
    verbose=0
).flatten()


# -------------------------------------------------
# 7. RESULTADOS FINAIS
# -------------------------------------------------

# Avalia a Regressão Logística com threshold 0.6
avaliar(
    "REGRESSÃO LOGÍSTICA - TESTE",
    y_teste,
    prob_regressao,
    threshold=0.6
)

# Avalia a MLP 16-8 com threshold 0.6
avaliar(
    "MLP 16-8 - TESTE",
    y_teste,
    prob_mlp,
    threshold=0.6
)

# Confirma o encerramento do procedimento
print("\nAvaliação final concluída.")

# Reforça que não devemos ajustar o modelo com base no teste
print("Os resultados do teste não devem ser usados para novos ajustes.")
