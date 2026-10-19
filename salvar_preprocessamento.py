
# Importa o Joblib para salvar objetos do Python
import joblib

# Importa Path para trabalhar com diretórios
from pathlib import Path

# Importa nossa função de preparação dos dados
from preparacao_dados import preparar_dados


# Prepara os dados e recupera os transformadores ajustados
(
    X_treino,
    y_treino,
    X_val,
    y_val,
    X_teste,
    y_teste,
    transformadores
) = preparar_dados(retornar_transformadores=True)


# Define a pasta onde os arquivos serão armazenados
pasta_modelos = Path("modelos")

# Cria a pasta caso ela não exista
pasta_modelos.mkdir(exist_ok=True)

# Define o caminho do arquivo de pré-processamento
caminho = pasta_modelos / "preprocessamento.joblib"

# Salva os transformadores e a ordem das colunas
joblib.dump(transformadores, caminho)

# Recupera o arquivo para verificar se foi salvo corretamente
transformadores_carregados = joblib.load(caminho)

# Mostra o resultado da operação
print("\n========== PRÉ-PROCESSAMENTO SALVO ==========")

# Mostra o local do arquivo
print("Arquivo:", caminho)

# Mostra a quantidade de características originais
print(
    "Características originais:",
    len(transformadores_carregados["colunas_originais"])
)

# Mostra a quantidade de entradas após o pré-processamento
print("Entradas da MLP:", X_treino.shape[1])

# Confirma o encerramento
print("\nPré-processamento salvo com sucesso!")
