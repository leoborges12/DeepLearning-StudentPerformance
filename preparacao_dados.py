
# Importa o Pandas para manipular os dados
import pandas as pd

# Importa o NumPy para trabalhar com matrizes
import numpy as np

# Importa a função para dividir os dados
from sklearn.model_selection import train_test_split

# Importa os transformadores dos dados
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Prepara os dados e opcionalmente retorna os transformadores
def preparar_dados(retornar_transformadores=False):

    # Lê o arquivo CSV de matemática
    dados = pd.read_csv("dados/student-mat.csv", sep=";")

    # Cria a classe 1 para aprovado e 0 para reprovado
    dados["aprovado"] = (dados["G3"] >= 10).astype(int)

    # Separa a variável que queremos prever
    y = dados["aprovado"]

    # Remove as notas e a variável-alvo das entradas
    X = dados.drop(columns=["G1", "G2", "G3", "aprovado"])

    # Identifica os nomes das colunas categóricas
    colunas_categoricas = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Identifica os nomes das colunas numéricas
    colunas_numericas = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Exibe as dimensões originais
    print("Dimensões de X:", X.shape)
    print("Dimensões de y:", y.shape)

    # Exibe as características categóricas
    print("\nColunas categóricas:")
    print(colunas_categoricas)
    print("Quantidade:", len(colunas_categoricas))

    # Exibe as características numéricas
    print("\nColunas numéricas:")
    print(colunas_numericas)
    print("Quantidade:", len(colunas_numericas))

    # Separa 15% dos alunos para o teste final
    X_dev, X_teste, y_dev, y_teste = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y
    )

    # Divide os alunos restantes em treinamento e validação
    X_treino, X_val, y_treino, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=59,
        random_state=42,
        stratify=y_dev
    )

    # Exibe as dimensões antes das transformações
    print("\nTreinamento:", X_treino.shape)
    print("Validação:", X_val.shape)
    print("Teste:", X_teste.shape)

    # Cria o codificador das características categóricas
    codificador = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    # Aprende as categorias somente no treinamento
    X_treino_cat = codificador.fit_transform(
        X_treino[colunas_categoricas]
    )

    # Transforma a validação usando o codificador aprendido
    X_val_cat = codificador.transform(
        X_val[colunas_categoricas]
    )

    # Transforma o teste usando o mesmo codificador
    X_teste_cat = codificador.transform(
        X_teste[colunas_categoricas]
    )

    # Cria o padronizador das características numéricas
    padronizador = StandardScaler()

    # Aprende média e desvio-padrão somente no treinamento
    X_treino_num = padronizador.fit_transform(
        X_treino[colunas_numericas]
    )

    # Padroniza a validação com os valores do treinamento
    X_val_num = padronizador.transform(
        X_val[colunas_numericas]
    )

    # Padroniza o teste com os mesmos valores
    X_teste_num = padronizador.transform(
        X_teste[colunas_numericas]
    )

    # Mostra as dimensões de cada transformação
    print("\nFormato após One-Hot Encoding:", X_treino_cat.shape)
    print("Formato após padronização:", X_treino_num.shape)

    # Junta as características categóricas e numéricas
    X_treino_final = np.concatenate(
        [X_treino_cat, X_treino_num],
        axis=1
    ).astype(np.float32)

    # Faz a mesma junção para a validação
    X_val_final = np.concatenate(
        [X_val_cat, X_val_num],
        axis=1
    ).astype(np.float32)

    # Faz a mesma junção para o teste
    X_teste_final = np.concatenate(
        [X_teste_cat, X_teste_num],
        axis=1
    ).astype(np.float32)

    # Exibe as dimensões finais
    print("\n--- MATRIZES FINAIS ---")
    print("Treinamento:", X_treino_final.shape)
    print("Validação:", X_val_final.shape)
    print("Teste:", X_teste_final.shape)

    # Organiza os seis valores utilizados pelos códigos anteriores
    conjuntos = (
        X_treino_final,
        y_treino.to_numpy(dtype=np.int32),
        X_val_final,
        y_val.to_numpy(dtype=np.int32),
        X_teste_final,
        y_teste.to_numpy(dtype=np.int32)
    )

    # Verifica se precisamos devolver os transformadores
    if retornar_transformadores:

        # Guarda os transformadores e a ordem original das colunas
        transformadores = {
            "codificador": codificador,
            "padronizador": padronizador,
            "colunas_categoricas": colunas_categoricas,
            "colunas_numericas": colunas_numericas,
            "colunas_originais": X.columns.tolist()
        }

        # Retorna os seis conjuntos mais os transformadores
        return (*conjuntos, transformadores)

    # Mantém o retorno original para os outros arquivos
    return conjuntos
