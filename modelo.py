
# Importa o TensorFlow para criar a rede neural
import tensorflow as tf


# Função que cria uma MLP com arquitetura configurável
def criar_modelo(quantidade_entradas, neuronios_1=16, neuronios_2=8):

    # Cria uma rede neural sequencial
    modelo = tf.keras.Sequential([

        # Define a quantidade de características de entrada
        tf.keras.Input(shape=(quantidade_entradas,)),

        # Primeira camada oculta com quantidade configurável de neurônios
        tf.keras.layers.Dense(
            neuronios_1,
            activation="relu"
        ),

        # Segunda camada oculta com quantidade configurável de neurônios
        tf.keras.layers.Dense(
            neuronios_2,
            activation="relu"
        ),

        # Camada de saída com um neurônio e Sigmoid
        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    # Compila a rede neural para classificação binária
    modelo.compile(

        # Utiliza o otimizador Adam
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        # Utiliza Binary Cross-Entropy como função de perda
        loss="binary_crossentropy",

        # Acompanha a acurácia durante o treinamento
        metrics=["accuracy"]
    )

    # Retorna o modelo compilado
    return modelo
