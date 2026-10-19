
# Importa o NumPy para cálculos matemáticos
import numpy as np

# Importa o TensorFlow para treinar as redes neurais
import tensorflow as tf

# Importa a regressão logística
from sklearn.linear_model import LogisticRegression

# Importa as métricas de classificação
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)

# Importa os pesos de classe
from sklearn.utils.class_weight import compute_class_weight

# Importa o Early Stopping
from tensorflow.keras.callbacks import EarlyStopping

# Importa a preparação dos dados
from preparacao_dados import preparar_dados

# Importa a criação das redes neurais
from modelo import criar_modelo


# -------------------------------------------------
# 1. FUNÇÃO DE CÁLCULO DAS MÉTRICAS
# -------------------------------------------------

# Calcula as métricas utilizando respostas e previsões
def calcular_metricas(y_real, previsoes):

    # Devolve as métricas em um dicionário
    return {
        "acuracia": accuracy_score(y_real, previsoes),

        "recall": recall_score(
            y_real, previsoes,
            pos_label=0,
            zero_division=0
        ),

        "precision": precision_score(
            y_real, previsoes,
            pos_label=0,
            zero_division=0
        ),

        "f1": f1_score(
            y_real, previsoes,
            pos_label=0,
            zero_division=0
        )
    }


# -------------------------------------------------
# 2. PREPARAÇÃO DOS DADOS
# -------------------------------------------------

# Prepara os mesmos dados utilizados anteriormente
X_treino, y_treino, X_val, y_val, X_teste, y_teste = preparar_dados()

# Define o threshold único da comparação
threshold = 0.6

# Identifica as classes existentes no treinamento
classes = np.unique(y_treino)

# Calcula os pesos balanceados das classes
pesos = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_treino
)

# Organiza os pesos em um dicionário
pesos_classes = {
    int(classe): float(peso)
    for classe, peso in zip(classes, pesos)
}


# -------------------------------------------------
# 3. REGRESSÃO LOGÍSTICA
# -------------------------------------------------

# Cria a regressão logística com pesos balanceados
regressao = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# Treina o modelo clássico
regressao.fit(X_treino, y_treino)

# Encontra a coluna correspondente à classe aprovado
indice_aprovado = list(regressao.classes_).index(1)

# Calcula as probabilidades estimadas de aprovação
probabilidades = regressao.predict_proba(X_val)[:, indice_aprovado]

# Converte probabilidades em classificações
previsoes = (probabilidades >= threshold).astype(int)

# Calcula as métricas da regressão logística
metricas_regressao = calcular_metricas(y_val, previsoes)


# -------------------------------------------------
# 4. EXPERIMENTOS COM AS REDES NEURAIS
# -------------------------------------------------

# Define as arquiteturas que vamos comparar
arquiteturas = [
    ("MLP 16-8", 16, 8),
    ("MLP 8-4", 8, 4)
]

# Define as sementes dos experimentos
sementes = [42, 123, 456]

# Cria um dicionário para armazenar os resultados
resultados_mlp = {
    "MLP 16-8": [],
    "MLP 8-4": []
}

# Percorre as arquiteturas
for nome, neuronios_1, neuronios_2 in arquiteturas:

    # Percorre as sementes
    for semente in sementes:

        # Limpa o estado anterior do TensorFlow
        tf.keras.backend.clear_session()

        # Define a semente deste experimento
        tf.keras.utils.set_random_seed(semente)

        # Cria a rede neural com a arquitetura atual
        modelo = criar_modelo(
            X_treino.shape[1],
            neuronios_1,
            neuronios_2
        )

        # Configura a parada antecipada
        parada = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )

        # Treina a rede neural
        modelo.fit(
            X_treino,
            y_treino,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            class_weight=pesos_classes,
            callbacks=[parada],
            verbose=0
        )

        # Calcula as probabilidades de aprovação
        probabilidades = modelo.predict(
            X_val,
            verbose=0
        ).flatten()

        # Converte as probabilidades em classes
        previsoes = (probabilidades >= threshold).astype(int)

        # Calcula as métricas desta execução
        metricas = calcular_metricas(y_val, previsoes)

        # Armazena as métricas no modelo correspondente
        resultados_mlp[nome].append(metricas)


# -------------------------------------------------
# 5. RESULTADOS COMPARATIVOS
# -------------------------------------------------

# Exibe o resultado da regressão logística
print("\n========== REGRESSÃO LOGÍSTICA ==========")
print(f"Acurácia: {metricas_regressao['acuracia'] * 100:.2f}%")
print(f"Recall: {metricas_regressao['recall'] * 100:.2f}%")
print(f"Precision: {metricas_regressao['precision'] * 100:.2f}%")
print(f"F1-score: {metricas_regressao['f1'] * 100:.2f}%")


# Percorre os resultados das duas MLPs
for nome, resultados in resultados_mlp.items():

    # Mostra o nome do modelo
    print(f"\n========== {nome} ==========")

    # Percorre as métricas
    for metrica in ["acuracia", "recall", "precision", "f1"]:

        # Recupera os valores das três execuções
        valores = [resultado[metrica] for resultado in resultados]

        # Calcula a média dos resultados
        media = np.mean(valores)

        # Calcula o desvio-padrão populacional
        desvio = np.std(valores)

        # Mostra os resultados
        print(
            f"{metrica}: {media * 100:.2f}% "
            f"(desvio: {desvio * 100:.2f} p.p.)"
        )


# -------------------------------------------------
# 6. SELEÇÃO PRELIMINAR
# -------------------------------------------------

# Calcula o F1-score médio de cada arquitetura MLP
f1_mlp_16 = np.mean([
    resultado["f1"]
    for resultado in resultados_mlp["MLP 16-8"]
])

f1_mlp_8 = np.mean([
    resultado["f1"]
    for resultado in resultados_mlp["MLP 8-4"]
])

# Organiza os resultados para comparação
pontuacoes = {
    "Regressão Logística": metricas_regressao["f1"],
    "MLP 16-8": f1_mlp_16,
    "MLP 8-4": f1_mlp_8
}

# Seleciona o maior F1-score observado na validação
melhor_modelo = max(pontuacoes, key=pontuacoes.get)

# Exibe a escolha preliminar
print("\n========== SELEÇÃO PRELIMINAR ==========")
print("Modelo com maior F1:", melhor_modelo)
print(f"F1 obtido: {pontuacoes[melhor_modelo] * 100:.2f}%")

# Reforça que os dados de teste ainda não foram utilizados
print("\nConjunto de teste permanece reservado.")
