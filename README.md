# Análise de candidatos a vereador — Eleições Municipais de 2024 na Bahia

Projeto desenvolvido para o desafio técnico de Analista de Dados.

O objetivo é analisar o perfil dos candidatos a vereador na Bahia e verificar quais características diferenciam os candidatos eleitos dos não eleitos. O projeto reúne tratamento de dados em Python, análise estatística, um modelo preditivo exploratório e um painel em Power BI.

## Dados utilizados

Foram utilizados dois arquivos públicos do TSE:

- `consulta_cand_2024_BA.csv`: dados dos candidatos;
- `bem_candidato_2024_BA.csv`: bens declarados pelos candidatos.

A chave de ligação entre os arquivos é `SQ_CANDIDATO`.

Por questões de tamanho e de organização, os arquivos originais não são versionados no repositório. Eles devem ser colocados em:

```text
data/raw/
├── consulta_cand_2024_BA.csv
└── bem_candidato_2024_BA.csv
```

## Tecnologias utilizadas

- Python
- pandas
- NumPy
- Matplotlib
- SciPy
- scikit-learn
- Power BI Desktop
- DAX
- Git e GitHub

## Estrutura esperada do projeto

```text
.
├── analise_vereadores.py
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── consulta_cand_2024_BA.csv
│   │   └── bem_candidato_2024_BA.csv
│   └── vereadores_ba_2024_powerbi.csv
├── outputs/
├── figures/
└── powerbi/
    └── dashboard_eleicoes_ba_2024.pbix
```

## Como executar

### 1. Criar e ativar o ambiente virtual

No Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

No Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar as dependências

Com o arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

Ou diretamente:

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

### 3. Colocar os arquivos do TSE em `data/raw/`

Os nomes esperados são:

```text
consulta_cand_2024_BA.csv
bem_candidato_2024_BA.csv
```

### 4. Executar a análise

```bash
python analise_vereadores.py
```

O script cria automaticamente as pastas de saída e gera:

- base tratada para o Power BI;
- tabelas-resumo em CSV;
- gráficos em PNG;
- testes qui-quadrado;
- resumo executivo;
- métricas e gráficos da regressão logística.

## O que o código em Python faz

O fluxo principal é:

1. carrega os arquivos do TSE;
2. trata codificação, separador e valores ausentes;
3. filtra apenas candidatos a vereador;
4. identifica eleitos, não eleitos e situações não informadas;
5. converte valores monetários brasileiros;
6. soma os bens por candidato;
7. calcula idade e cria faixas etárias;
8. cria faixas de patrimônio;
9. calcula indicadores e tabelas por gênero, escolaridade, idade, partido, ocupação e patrimônio;
10. executa testes qui-quadrado;
11. treina uma regressão logística exploratória;
12. exporta os resultados e a base usada no Power BI.

## Principais dificuldades e decisões no Python

### Valores nulos

Os arquivos usam diferentes códigos para ausência de informação, como `#NULO`, `#NE` e `NÃO INFORMADO`. Esses valores foram convertidos para nulos reais do pandas.

A situação eleitoral ausente não foi classificada automaticamente como derrota. Ela permanece como valor ausente e não entra no cálculo da taxa principal de eleição.

### Codificação e separador

Os arquivos do TSE foram lidos com:

```python
encoding="latin-1"
sep=";"
```

Isso evita problemas de acentuação e de separação das colunas.

### Valores monetários brasileiros

Valores como `25.000,50` chegam como texto. O código remove o separador de milhar, troca a vírgula decimal por ponto e converte o resultado para número.

### Uso da estatística sem dificultar a leitura

O projeto usa medidas como proporção, mediana, quartis e testes de associação, mas apresenta os resultados com nomes e frases interpretáveis.

A mediana foi priorizada na comparação de patrimônio porque poucos valores muito altos podem distorcer a média.

### Faixas de patrimônio

As faixas foram criadas para facilitar a comparação no dashboard. Elas não representam uma classificação oficial do TSE, mas uma escolha analítica documentada.

### Uso responsável de IA

A IA foi usada como ferramenta de apoio para revisão e orientação, correção, ideias.

Foram adotados limites de uso:

- resultados verificados;
- decisões de limpeza revisados;
- usado sugestões de métricas junto com o pedido no desafio;
- foi conferido quais alguns rumos tomar;
- o modelo e suas limitações foram documentados.



## Teste qui-quadrado

O teste qui-quadrado foi usado para verificar se duas variáveis categóricas apresentam associação.

Exemplo:

- variável 1: gênero;
- variável 2: eleito ou não eleito.

A hipótese nula afirma que as variáveis são independentes. Um valor-p pequeno indica evidência contra essa hipótese.

Neste projeto, o teste foi aplicado a variáveis como:

- gênero;
- escolaridade;
- faixa etária;
- faixa de patrimônio;
- partido.

O teste responde se existe associação estatística, mas não informa sozinho:

- a intensidade da associação;
- qual grupo é mais importante;
- se existe relação de causa e efeito.

Como a base possui muitos registros, valores-p muito pequenos podem ocorrer mesmo quando a diferença prática não é grande. Por isso, o teste foi interpretado junto com taxas, quantidades e gráficos.

## Regressão logística

A regressão logística foi escolhida porque o resultado a prever é binário:

- `0`: não eleito;
- `1`: eleito.

O modelo estima a probabilidade de eleição com base em características disponíveis antes ou durante o registro da candidatura.

Foram utilizadas variáveis como:

- idade;
- patrimônio declarado em escala logarítmica;
- quantidade de bens;
- gênero;
- escolaridade;
- cor ou raça;
- ocupação;
- partido.

### Preparação do modelo

O processo inclui:

- separação de 80% dos dados para treino e 20% para teste;
- divisão estratificada para preservar a proporção de eleitos;
- preenchimento de nulos;
- padronização das variáveis numéricas;
- transformação das categorias com one-hot encoding;
- balanceamento das classes com `class_weight="balanced"`.

### Avaliação

O modelo é avaliado por:

- matriz de confusão;
- precisão;
- recall;
- F1-score;
- curva ROC;
- ROC AUC.

Na matriz de confusão:

- **verdadeiro positivo:** eleito previsto corretamente como eleito;
- **verdadeiro negativo:** não eleito previsto corretamente como não eleito;
- **falso positivo:** não eleito previsto incorretamente como eleito;
- **falso negativo:** eleito previsto incorretamente como não eleito.

Os coeficientes positivos indicam associação com maior probabilidade estimada de eleição. Os coeficientes negativos indicam associação com menor probabilidade estimada.

Esses coeficientes mostram contribuição preditiva, não causalidade. O modelo é exploratório e não deve ser usado para decidir quem deveria ou não ser eleito.

## Power BI

O Power BI foi usado para transformar a análise em um painel dinâmica e resumido

### Principais recursos

- cartões com total de candidatos, eleitos, taxa de eleição e patrimônio mediano;
- filtros por município, partido e situação eleitoral;
- comparações por gênero, idade, escolaridade, ocupação e patrimônio;
- medidas em DAX;
- comparação entre eleitos e não eleitos.

### Decisões visuais

O espaço do painel era limitado, então foram priorizados poucos gráficos com maior capacidade de explicar a pergunta central.

Também foram adotadas as seguintes escolhas:

- fundo claro e discreto;
- Usado cons pastéis 
- cores diferentes entre gráficos e elementos;
- transparência excessiva para mostrar dados suspensos e simples e agradáveis ao ser humano;
- percentuais sempre do lado de suas barras representativas;


## Limitações

- Os dados representam as eleições municipais de 2024 na Bahia.
- Patrimônio declarado não corresponde necessariamente à riqueza real do candidato.
- Ocupação declarada como “vereador” pode refletir vantagem de incumbência, mas a base utilizada não comprova sozinha uma candidatura à reeleição.
- Associação estatística não significa causalidade.
- O modelo preditivo é exploratório e depende das variáveis disponíveis.
- Resultados sem situação eleitoral informada foram preservados, mas excluídos das análises que exigem resultado conhecido.
- Power bi desktop

## Como abrir o painel

Abra o arquivo `.pbix` no Power BI Desktop.

Caso o Power BI solicite uma nova localização para a fonte, selecione:

```text
data/vereadores_ba_2024_powerbi.csv
```

Depois, clique em **Atualizar**.

Talvez ocorra erros de incrogruência dos dados depois da revisão do código python
Foi deixado código inicial nesse github mostrando comentários informais para estímulo no desenvolvimento
