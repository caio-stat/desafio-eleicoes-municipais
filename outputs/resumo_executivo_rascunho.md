# Resumo executivo (Rascunho)

A análise avaliou 32.879 candidaturas a vereador
nas eleições municipais de 2024 na Bahia. Deste total,
31.623 possuíam situação eleitoral
informada e 1.256 estavam sem resultado
preenchido na base utilizada.

## Principais números

- Total de eleitos: 4.591
- Taxa de eleição entre candidaturas com resultado informado: 14,52%
- Intervalo binomial exploratório de 95%: 14,13% a 14,91%
- Patrimônio mediano declarado dos eleitos: R$ 81.886,00
- Patrimônio mediano declarado dos não eleitos: R$ 4.500,00
- Eleitos sem bens declarados: 21,19%
- Não eleitos sem bens declarados: 42,81%
- ROC AUC da regressão logística exploratória: 0.814

## Interpretação

Os dados mostram diferenças entre eleitos e não eleitos em características
como patrimônio declarado, gênero, escolaridade, faixa etária, partido e
ocupação. O patrimônio foi resumido principalmente pela mediana, pois sua
distribuição é assimétrica e contém valores extremos que influenciam fortemente
a média.

Os testes qui-quadrado foram usados para avaliar associação entre variáveis
categóricas e o resultado eleitoral. A regressão logística foi incluída como
análise preditiva exploratória, com divisão entre treino e teste. Seus
coeficientes e métricas indicam capacidade de diferenciação entre os grupos,
mas não devem ser interpretados como relações de causa e efeito.

## Decisões metodológicas

- Foram classificados como eleitos os registros cuja situação começa com a palavra "ELEITO".
- Situações eleitorais ausentes foram preservadas e excluídas das comparações entre eleitos e não eleitos.
- Os bens foram somados por candidato usando a chave `SQ_CANDIDATO`.
- A variável de patrimônio usada no modelo foi transformada por `log1p` devido à forte assimetria e à presença de zeros.