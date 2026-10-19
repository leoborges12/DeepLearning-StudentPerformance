
# Importa o Pandas para manipular os dados
import pandas as pd

# Importa o NumPy para trabalhar com matrizes
import numpy as np

# Importa o Joblib para carregar o pré-processamento
import joblib

# Importa o TensorFlow para carregar a rede neural
import tensorflow as tf

# Importa Path para localizar os arquivos
from pathlib import Path


# -------------------------------------------------
# 1. CARREGAMENTO DOS ARQUIVOS
# -------------------------------------------------

# Define a pasta dos modelos
pasta = Path("modelos")

# Carrega o pré-processamento previamente salvo
transformadores = joblib.load(
    pasta / "preprocessamento.joblib"
)

# Carrega a MLP treinada sem realizar um novo treinamento
modelo = tf.keras.models.load_model(
    pasta / "mlp_student_performance.keras"
)

# Confirma o carregamento dos arquivos
print("\nArquivos carregados com sucesso!")


# -------------------------------------------------
# 2. LEITURA DE UM EXEMPLO
# -------------------------------------------------

# Lê o dataset original
dados = pd.read_csv(
    "dados/student-mat.csv",
    sep=";"
)

# Seleciona um estudante para demonstrar o funcionamento
estudante = dados.iloc[[0]].copy()

# Recupera a ordem das características originais
colunas_originais = transformadores["colunas_originais"]

# Mantém somente as características utilizadas pela MLP
X_novo = estudante[colunas_originais]


# -------------------------------------------------
# 3. PRÉ-PROCESSAMENTO
# -------------------------------------------------

# Recupera o codificador das categorias
codificador = transformadores["codificador"]

# Recupera o padronizador das características numéricas
padronizador = transformadores["padronizador"]

# Recupera os nomes das colunas categóricas
colunas_cat = transformadores["colunas_categoricas"]

# Recupera os nomes das colunas numéricas
colunas_num = transformadores["colunas_numericas"]

# Aplica o One-Hot Encoder já treinado
X_cat = codificador.transform(
    X_novo[colunas_cat]
)

# Aplica o StandardScaler já treinado
X_num = padronizador.transform(
    X_novo[colunas_num]
)

# Junta as características na mesma ordem do treinamento
X_final = np.concatenate(
    [X_cat, X_num],
    axis=1
).astype(np.float32)

# Confere o número de características
print("Formato da entrada:", X_final.shape)

# Garante que o modelo recebe exatamente 56 entradas
assert X_final.shape[1] == 56


# -------------------------------------------------
# 4. PREVISÃO
# -------------------------------------------------

# Obtém a probabilidade estimada de aprovação
probabilidade = float(
    modelo.predict(X_final, verbose=0)[0][0]
)

# Define o threshold escolhido anteriormente
threshold = 0.6

# Converte a probabilidade em uma classificação
classe_prevista = int(probabilidade >= threshold)

# Mostra a probabilidade estimada de aprovação
print(f"\nProbabilidade de aprovação: {probabilidade * 100:.2f}%")

# Verifica a classificação prevista
if classe_prevista == 1:

    # Mostra que a rede classificou o aluno como aprovado
    print("Classificação prevista: APROVADO")

else:

    # Mostra que a rede classificou o aluno como reprovado
    print("Classificação prevista: REPROVADO")

# Confirma que nenhum treinamento foi realizado
print("\nPrevisão realizada sem treinar novamente a rede!")
