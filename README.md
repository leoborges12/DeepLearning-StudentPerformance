# DeepLearning — Student Performance

Projeto de classificação do desempenho acadêmico de estudantes utilizando **Deep Learning com redes neurais artificiais**, desenvolvido em Python com TensorFlow, Keras e Scikit-learn.

## 1. Objetivo

O objetivo deste projeto é investigar a utilização de redes neurais artificiais para identificar estudantes com possível risco de reprovação na disciplina de Matemática, considerando características pessoais, familiares, escolares e comportamentais.

Além das redes neurais, foi implementada uma Regressão Logística para comparação com uma abordagem clássica de Machine Learning.

O projeto tem finalidade educacional e experimental, não sendo destinado à tomada automatizada de decisões acadêmicas.

## 2. Dataset

Foi utilizado o conjunto de dados **Student Performance**, disponibilizado pelo UCI Machine Learning Repository.

Fonte: https://archive.ics.uci.edu/dataset/320/student+performance

O conjunto de Matemática (`student-mat.csv`) contém 395 estudantes e 33 atributos originais.

A classificação foi definida a partir da nota final `G3`:

- **Classe 0 — Reprovado:** nota final inferior a 10.
- **Classe 1 — Aprovado:** nota final maior ou igual a 10.

As notas `G1`, `G2` e `G3` foram removidas das características de entrada, permitindo investigar previsões sem utilizar essas notas como informações do modelo.

Após a seleção das características, permaneceram 30 variáveis de entrada.

## 3. Preparação dos dados

O pré-processamento foi realizado com Pandas e Scikit-learn.

Foram utilizadas as seguintes técnicas:

- **One-Hot Encoding:** transforma características categóricas em representações numéricas.
- **StandardScaler:** padroniza características numéricas.
- **Divisão estratificada:** preserva aproximadamente a proporção de estudantes aprovados e reprovados.

A divisão resultou em:

| Conjunto | Estudantes |
|---|---:|
| Treinamento | 276 |
| Validação | 59 |
| Teste | 60 |
| **Total** | **395** |

Os transformadores foram ajustados somente com os dados de treinamento, reduzindo o risco de vazamento de informações.

Após o pré-processamento, cada estudante passou a ser representado por **56 características numéricas**.

## 4. Redes neurais implementadas

Foram estudadas duas arquiteturas de redes neurais do tipo Multilayer Perceptron (MLP).

**MLP 16–8**

- Camada de entrada: 56 características.
- Primeira camada oculta: 16 neurônios com ReLU.
- Segunda camada oculta: 8 neurônios com ReLU.
- Camada de saída: 1 neurônio com Sigmoid.
- Parâmetros treináveis: 1.057.

**MLP 8–4**

- Camada de entrada: 56 características.
- Primeira camada oculta: 8 neurônios com ReLU.
- Segunda camada oculta: 4 neurônios com ReLU.
- Camada de saída: 1 neurônio com Sigmoid.
- Parâmetros treináveis: 497.

### Configurações do treinamento

- Otimizador: Adam.
- Taxa de aprendizado: 0,001.
- Função de perda: Binary Crossentropy.
- Batch size: 32.
- Número máximo de épocas: 100.
- Early Stopping: monitoramento de `val_loss`, paciência de 10 épocas e restauração dos melhores pesos.
- Class Weights: balanceamento da importância das classes.
- Threshold de classificação: 0,6.

## 5. Comparação das arquiteturas

As duas redes foram avaliadas utilizando as sementes aleatórias 42, 123 e 456, mantendo os mesmos conjuntos de treinamento e validação.

| Métrica média | MLP 16–8 | MLP 8–4 |
|---|---:|---:|
| Acurácia | 55,93% | 51,98% |
| Recall dos reprovados | 63,16% | 68,42% |
| Precision dos reprovados | 38,48% | 36,96% |
| F1-score dos reprovados | 47,67% | 47,85% |
| Desvio-padrão do F1 | 5,29 p.p. | 3,71 p.p. |

Os resultados indicaram desempenho médio semelhante entre as duas arquiteturas.

Como foram utilizadas somente três sementes e um conjunto pequeno de validação, os resultados não permitem concluir que uma arquitetura seja consistentemente superior à outra.

## 6. Comparação com Regressão Logística

Também foi implementado um classificador de Regressão Logística, utilizando:

- Os mesmos conjuntos de dados pré-processados.
- Pesos balanceados entre as classes.
- Threshold de 0,6.
- Máximo de 1.000 iterações.

Na validação, a Regressão Logística apresentou:

| Métrica | Resultado |
|---|---:|
| Acurácia | 62,71% |
| Recall dos reprovados | 63,16% |
| Precision dos reprovados | 44,44% |
| F1-score dos reprovados | 52,17% |

Esse resultado mostrou que um classificador mais simples também pode apresentar desempenho competitivo com redes neurais.

## 7. Avaliação final

Após a etapa exploratória, foram avaliadas configurações previamente definidas da Regressão Logística e da MLP 16–8 nos 60 estudantes do conjunto de teste.

| Métrica | Regressão Logística | MLP 16–8 |
|---|---:|---:|
| Acurácia | 56,67% | 60,00% |
| Recall dos reprovados | 55,00% | 70,00% |
| Precision dos reprovados | 39,29% | 43,75% |
| F1-score dos reprovados | 45,83% | 53,85% |

### Matriz de confusão — MLP 16–8

| Classe real / Prevista | Reprovado | Aprovado |
|---|---:|---:|
| Reprovado | 14 | 6 |
| Aprovado | 18 | 22 |

A rede neural identificou corretamente 14 dos 20 estudantes reprovados.

Entretanto, também classificou incorretamente 18 estudantes aprovados como reprovados.

No conjunto de teste, um baseline que previsse aprovação para todos os estudantes alcançaria 66,67% de acurácia, mas não identificaria nenhum reprovado.

Assim, a principal vantagem observada na MLP foi sua capacidade de identificar parte dos estudantes em risco, e não sua acurácia geral.

A superioridade da MLP sobre a Regressão Logística nesse conjunto de teste não garante superioridade em outras amostras ou turmas.

## 8. Organização do projeto

```text
DeepLearning-StudentPerformance/
│
├── dados/
│   ├── student-mat.csv
│   └── student-por.csv
│
├── modelos/
│   ├── mlp_student_performance.keras
│   └── preprocessamento.joblib
│
├── preparacao_dados.py
├── modelo.py
├── graficos.py
├── avaliacao.py
├── baseline.py
├── main.py
├── comparar_modelos.py
├── comparar_regressao.py
├── selecao_modelo.py
├── avaliacao_final.py
├── salvar_preprocessamento.py
├── testar_modelo_salvo.py
└── README.md
```

### Principais arquivos

| Arquivo | Responsabilidade |
|---|---|
| `preparacao_dados.py` | Leitura, divisão e pré-processamento |
| `modelo.py` | Definição e compilação das MLPs |
| `main.py` | Treinamento e análise experimental |
| `graficos.py` | Visualização das curvas de treinamento |
| `avaliacao.py` | Cálculo das métricas de classificação |
| `baseline.py` | Avaliação do classificador majoritário |
| `comparar_modelos.py` | Comparação de arquiteturas e sementes |
| `comparar_regressao.py` | Experimento com Regressão Logística |
| `selecao_modelo.py` | Seleção preliminar dos modelos |
| `avaliacao_final.py` | Avaliação nos dados reservados e salvamento da MLP |
| `salvar_preprocessamento.py` | Persistência dos transformadores |
| `testar_modelo_salvo.py` | Carregamento da MLP e realização de inferência |

## 9. Tecnologias utilizadas

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib

## 10. Como executar

Os comandos abaixo consideram um terminal PowerShell no Windows.

### Criar um ambiente virtual

```powershell
python -m venv .venv
```

### Ativar o ambiente

```powershell
.\.venv\Scripts\Activate.ps1
```

### Instalar as dependências

```powershell
python -m pip install tensorflow scikit-learn pandas numpy matplotlib joblib
```

Para reproduzir exatamente o ambiente utilizado, recomenda-se instalar as versões registradas em um arquivo `requirements.txt`.

### Treinar a rede neural

```powershell
python main.py
```

### Comparar arquiteturas

```powershell
python comparar_modelos.py
```

### Comparar com Regressão Logística

```powershell
python comparar_regressao.py
```

### Executar a seleção preliminar

```powershell
python selecao_modelo.py
```

### Reproduzir a avaliação final

```powershell
python avaliacao_final.py
```

Essa execução treina novamente os modelos com as configurações documentadas. Os resultados do teste já foram registrados e não devem ser utilizados para novos ajustes.

### Salvar os transformadores

```powershell
python salvar_preprocessamento.py
```

### Carregar o modelo salvo e realizar uma previsão

```powershell
python testar_modelo_salvo.py
```

O carregamento utiliza os arquivos `.keras` e `.joblib`, sem executar um novo treinamento.

## 11. Limitações

Este projeto apresenta algumas limitações:

- O dataset possui apenas 395 registros.
- A avaliação final utilizou 60 estudantes.
- Foram utilizadas apenas três sementes aleatórias na comparação das arquiteturas.
- A exploração repetida do conjunto de validação pode ter influenciado a seleção das configurações.
- As probabilidades produzidas pela rede não passaram por uma avaliação específica de calibração.
- O modelo ainda apresenta falsos alertas e casos de reprovação não identificados.
- O sistema não foi validado em estudantes de outras instituições ou em novos períodos letivos.

Por isso, os resultados devem ser interpretados como um experimento acadêmico, não como comprovação de adequação para uso institucional.

## 12. Conclusão

Foi desenvolvido um fluxo de classificação de desempenho acadêmico utilizando redes neurais MLP e Regressão Logística.

O projeto contemplou preparação dos dados, treinamento, validação, comparação de arquiteturas, avaliação em conjunto de teste e persistência do modelo.

Na avaliação final, a MLP 16–8 obteve F1-score de 53,85% e Recall de 70,00% para os estudantes reprovados, superando a Regressão Logística nas métricas analisadas nesse conjunto.

Os experimentos demonstraram a importância de comparar diferentes algoritmos e de considerar métricas adequadas ao objetivo da classificação.

Como trabalhos futuros, destacam-se a utilização de conjuntos de dados maiores, a validação cruzada e a investigação de métodos para reduzir falsos alertas.

## 13. Referência do dataset

Cortez, P.; Silva, A. *Using Data Mining to Predict Secondary School Student Performance*. Proceedings of the 5th Future Business Technology Conference, 2008.

UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/320/student+performance
