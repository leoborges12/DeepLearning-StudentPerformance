
# Importa o NumPy para trabalhar com vetores e cálculos estatísticos
import numpy as np

# Importa o TensorFlow para executar as redes neurais
import tensorflow as tf

# Importa as métricas de classificação
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)

# Importa o cálculo dos pesos de classe
from sklearn.utils.class_weight import compute_class_weight

# Importa o mecanismo de parada antecipada
from tensorflow.keras.callbacks import EarlyStopping

# Importa nossa função de preparação dos dados
from preparacao_dados import preparar_dados

# Importa nossa função de criação da MLP
from modelo import criar_modelo


# -------------------------------------------------
# 1. FUNÇÃO QUE EXECUTA UM EXPERIMENTO
# -------------------------------------------------

# Recebe uma arquitetura, uma semente e os dados
def executar_experimento(
    nome,
    neuronios_1,
    neuronios_2,
    X_treino,
    y_treino,
    X_val,
    y_val,
    pesos_classes,
    semente=42
):

    # Limpa o estado anterior do TensorFlow
    tf.keras.backend.clear_session()

    # Define a semente utilizada neste treinamento
    tf.keras.utils.set_random_seed(semente)

    # Cria a arquitetura solicitada
    modelo = criar_modelo(
        X_treino.shape[1],
        neuronios_1,
        neuronios_2
    )

    # Cria o mecanismo de Early Stopping
    parada = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # Treina a rede com os dados de treinamento
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

    # Calcula as probabilidades de aprovação na validação
    probabilidades = modelo.predict(
        X_val,
        verbose=0
    ).flatten()

    # Mantém o mesmo threshold em todos os experimentos
    threshold = 0.6

    # Converte as probabilidades em classificações 0 ou 1
    previsoes = (probabilidades >= threshold).astype(int)

    # Calcula a acurácia geral
    acuracia = accuracy_score(y_val, previsoes)

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

    # Identifica a época com menor perda de validação
    melhor_epoca = int(np.argmin(historico.history["val_loss"]) + 1)

    # Exibe os resultados deste experimento
    print(f"\n--- {nome} | Semente {semente} ---")
    print("Parâmetros:", modelo.count_params())
    print("Épocas executadas:", len(historico.history["loss"]))
    print("Melhor época:", melhor_epoca)
    print(f"Acurácia: {acuracia * 100:.2f}%")
    print(f"Recall reprovados: {recall * 100:.2f}%")
    print(f"Precision reprovados: {precisao * 100:.2f}%")
    print(f"F1 reprovados: {f1 * 100:.2f}%")

    # Devolve os resultados para os cálculos estatísticos
    return {
        "modelo": nome,
        "semente": semente,
        "acuracia": acuracia,
        "recall": recall,
        "precisao": precisao,
        "f1": f1
    }


# -------------------------------------------------
# 2. FUNÇÃO PRINCIPAL
# -------------------------------------------------

def main():

    # Prepara os mesmos conjuntos utilizados anteriormente
    X_treino, y_treino, X_val, y_val, X_teste, y_teste = preparar_dados()

    # Identifica as classes no conjunto de treinamento
    classes = np.unique(y_treino)

    # Calcula os pesos das classes
    pesos = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_treino
    )

    # Cria o dicionário com os pesos
    pesos_classes = {
        int(classe): float(peso)
        for classe, peso in zip(classes, pesos)
    }

    # Define as três sementes do experimento
    sementes = [42, 123, 456]

    # Define as duas arquiteturas que vamos comparar
    arquiteturas = [
        ("MLP 16-8", 16, 8),
        ("MLP 8-4", 8, 4)
    ]

    # Cria uma lista para armazenar os resultados
    resultados = []

    # Percorre as sementes
    for semente in sementes:

        # Percorre as duas arquiteturas
        for nome, neuronios_1, neuronios_2 in arquiteturas:

            # Executa o experimento com a configuração atual
            resultado = executar_experimento(
                nome,
                neuronios_1,
                neuronios_2,
                X_treino,
                y_treino,
                X_val,
                y_val,
                pesos_classes,
                semente=semente
            )

            # Armazena o resultado obtido
            resultados.append(resultado)

    # -------------------------------------------------
    # 3. RESUMO ESTATÍSTICO
    # -------------------------------------------------

    # Exibe o cabeçalho do resumo final
    print("\n========== RESUMO DOS EXPERIMENTOS ==========")

    # Analisa separadamente cada arquitetura
    for nome, _, _ in arquiteturas:

        # Seleciona os resultados da arquitetura atual
        resultados_modelo = [
            resultado
            for resultado in resultados
            if resultado["modelo"] == nome
        ]

        # Percorre as métricas que queremos resumir
        print(f"\n--- {nome} ---")

        for metrica in ["acuracia", "recall", "precisao", "f1"]:

            # Obtém os valores dessa métrica nas três sementes
            valores = [
                resultado[metrica]
                for resultado in resultados_modelo
            ]

            # Calcula a média
            media = np.mean(valores)

            # Calcula o desvio-padrão populacional
            desvio = np.std(valores)

            # Mostra a média e o desvio-padrão
            print(
                f"{metrica}: "
                f"{media * 100:.2f}% "
                f"(desvio: {desvio * 100:.2f} p.p.)"
            )

    # Informa que os experimentos terminaram
    print("\nComparação concluída!")


# -------------------------------------------------
# 4. INÍCIO DO PROGRAMA
# -------------------------------------------------

# Executa a função principal somente ao iniciar este arquivo
if __name__ == "__main__":
    main()
