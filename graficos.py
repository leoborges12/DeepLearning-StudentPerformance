
# Importa o Matplotlib para construir gráficos
import matplotlib.pyplot as plt


# Função responsável por mostrar os gráficos de treinamento
def mostrar_graficos(historico):

    # Recupera os valores de acurácia do treinamento
    acuracia_treino = historico.history["accuracy"]

    # Recupera os valores de acurácia da validação
    acuracia_val = historico.history["val_accuracy"]

    # Recupera os valores da função de perda no treinamento
    perda_treino = historico.history["loss"]

    # Recupera os valores da função de perda na validação
    perda_val = historico.history["val_loss"]

    # Cria a sequência das épocas realmente executadas
    epocas = range(1, len(acuracia_treino) + 1)

    # Cria o primeiro gráfico
    plt.figure(figsize=(8, 5))

    # Desenha a acurácia do treinamento
    plt.plot(
        epocas,
        acuracia_treino,
        label="Treinamento"
    )

    # Desenha a acurácia da validação
    plt.plot(
        epocas,
        acuracia_val,
        label="Validação"
    )

    # Configura os elementos do gráfico
    plt.title("Acurácia da Rede Neural")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid(True)

    # Exibe o primeiro gráfico
    plt.show()

    # Cria o segundo gráfico
    plt.figure(figsize=(8, 5))

    # Desenha a perda do treinamento
    plt.plot(
        epocas,
        perda_treino,
        label="Treinamento"
    )

    # Desenha a perda da validação
    plt.plot(
        epocas,
        perda_val,
        label="Validação"
    )

    # Configura os elementos do gráfico
    plt.title("Função de Perda da Rede Neural")
    plt.xlabel("Épocas")
    plt.ylabel("Binary Cross-Entropy")
    plt.legend()
    plt.grid(True)

    # Exibe o segundo gráfico
    plt.show()
