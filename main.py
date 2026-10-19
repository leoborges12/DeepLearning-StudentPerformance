
# Importa o NumPy para trabalhar com matrizes
import numpy as np

# Importa o TensorFlow para construir e treinar a rede
import tensorflow as tf

# Importa a preparação dos dados
from preparacao_dados import preparar_dados

# Importa a criação da rede neural
from modelo import criar_modelo

# Importa a geração dos gráficos
from graficos import mostrar_graficos

# Importa a avaliação das classificações
from avaliacao import avaliar_classificacao

# Importa a avaliação do baseline
from baseline import avaliar_baseline

# Importa o cálculo de pesos das classes
from sklearn.utils.class_weight import compute_class_weight

# Importa o mecanismo de parada antecipada
from tensorflow.keras.callbacks import EarlyStopping


# -------------------------------------------------
# ETAPA 1: CONFIGURAÇÃO INICIAL
# -------------------------------------------------

# Define uma semente para melhorar a reprodutibilidade
tf.keras.utils.set_random_seed(42)


# -------------------------------------------------
# ETAPA 2: PREPARAÇÃO DOS DADOS
# -------------------------------------------------

# Carrega e prepara os dados dos estudantes
X_treino, y_treino, X_val, y_val, X_teste, y_teste = preparar_dados()

# Descobre quantas características cada aluno possui
quantidade_entradas = X_treino.shape[1]


# -------------------------------------------------
# ETAPA 3: CRIAÇÃO DA REDE NEURAL
# -------------------------------------------------

# Cria a MLP com as 56 características de entrada
modelo = criar_modelo(quantidade_entradas)

# Exibe a arquitetura e a quantidade de parâmetros
modelo.summary()


# -------------------------------------------------
# ETAPA 4: INSPEÇÃO DOS PESOS INICIAIS
# -------------------------------------------------

# Seleciona a primeira camada oculta
primeira_camada = modelo.layers[0]

# Obtém os pesos e os bias da primeira camada
pesos_iniciais, bias_iniciais = primeira_camada.get_weights()

# Exibe as dimensões dos parâmetros
print("\nDimensões dos pesos:", pesos_iniciais.shape)
print("Dimensões dos bias:", bias_iniciais.shape)

# Mostra os cinco primeiros pesos do primeiro neurônio
print("Pesos iniciais do primeiro neurônio:")
print(pesos_iniciais[:5, 0])

# Mostra o bias inicial do primeiro neurônio
print("Bias inicial:", bias_iniciais[0])


# -------------------------------------------------
# ETAPA 5: CÁLCULO DOS CLASS WEIGHTS
# -------------------------------------------------

# Identifica as classes existentes no treinamento
classes = np.unique(y_treino)

# Calcula os pesos de acordo com a frequência das classes
pesos_calculados = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_treino
)

# Converte os pesos em um dicionário
# A chave 0 representa reprovado
# A chave 1 representa aprovado
pesos_classes = {
    int(classe): float(peso)
    for classe, peso in zip(classes, pesos_calculados)
}

# Mostra os pesos utilizados durante o treinamento
print("\nPesos das classes:", pesos_classes)


# -------------------------------------------------
# ETAPA 6: CONFIGURAÇÃO DO EARLY STOPPING
# -------------------------------------------------

# Cria o mecanismo de parada antecipada
parada_antecipada = EarlyStopping(

    # Monitora a perda calculada nos dados de validação
    monitor="val_loss",

    # Interrompe após 10 épocas consecutivas sem melhora
    patience=10,

    # Restaura os melhores pesos encontrados na validação
    restore_best_weights=True
)


# -------------------------------------------------
# ETAPA 7: TREINAMENTO DA REDE NEURAL
# -------------------------------------------------

# Treina a rede neural utilizando os dados de treinamento
historico = modelo.fit(

    # Características dos alunos de treinamento
    X_treino,

    # Respostas verdadeiras dos alunos
    y_treino,

    # Define o limite máximo de épocas
    epochs=100,

    # Divide os alunos em mini-batches
    batch_size=32,

    # Acompanha o desempenho nos dados de validação
    validation_data=(X_val, y_val),

    # Aplica pesos diferentes às classes
    class_weight=pesos_classes,

    # Ativa a parada antecipada
    callbacks=[parada_antecipada],

    # Exibe o progresso das épocas
    verbose=1
)

# Mostra quantas épocas foram executadas
print("\nÉpocas executadas:", len(historico.history["loss"]))

# Mostra a época com a menor perda de validação
melhor_epoca = np.argmin(historico.history["val_loss"]) + 1

# Exibe a melhor época encontrada
print("Melhor época pela perda de validação:", melhor_epoca)


# -------------------------------------------------
# ETAPA 8: GRÁFICOS DO TREINAMENTO
# -------------------------------------------------

# Mostra os gráficos de acurácia e perda
mostrar_graficos(historico)


# -------------------------------------------------
# ETAPA 9: AVALIAÇÃO COM DIFERENTES THRESHOLDS
# -------------------------------------------------

# Define os thresholds que queremos investigar
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Percorre os thresholds definidos
for threshold in thresholds:

    # Avalia a MLP utilizando os alunos de validação
    avaliar_classificacao(
        modelo,
        X_val,
        y_val,
        threshold=threshold
    )


# -------------------------------------------------
# ETAPA 10: AVALIAÇÃO DO BASELINE
# -------------------------------------------------

# Avalia o classificador que sempre prevê aprovado
# Executa somente uma vez
avaliar_baseline(y_val)


# -------------------------------------------------
# ETAPA 11: ENCERRAMENTO
# -------------------------------------------------

# Informa que todas as etapas foram executadas
print("\nExecução do projeto concluída!")
