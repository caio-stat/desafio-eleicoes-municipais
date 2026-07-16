"""Análise dos candidatos a vereador nas eleições municipais de 2024 na Bahia.

O script segue um fluxo único e reproduzível:
1. carrega os microdados do TSE;
2. filtra as candidaturas a vereador;
3. trata a situação eleitoral e o patrimônio declarado;
4. cria variáveis de perfil;
5. produz tabelas, gráficos e testes estatísticos;
6. treina uma regressão logística exploratória;
7. exporta a base preparada para o Power BI e um resumo executivo.
"""

from pathlib import Path
from textwrap import dedent

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Usa um modo de renderização que salva os gráficos em arquivo sem abrir janelas.
# Isso permite executar o script também em terminal ou em ambientes sem interface gráfica.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Todos os caminhos partem da pasta onde este script está salvo.
# Assim, o projeto funciona mesmo quando executado em computadores diferentes.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures"

ARQUIVO_CANDIDATOS = DATA_RAW_DIR / "consulta_cand_2024_BA.csv"
ARQUIVO_BENS = DATA_RAW_DIR / "bem_candidato_2024_BA.csv"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# O TSE usa diferentes textos para representar informação ausente.
# Ao informar essa lista ao pandas, todos eles são convertidos para valores nulos reais.
VALORES_NULOS = [
    "#NULO",
    "#NE",
    "NÃO INFORMADO",
    "NAO INFORMADO",
    "",
    "NULO",
    "N/A",
    "NA",
    "NULL",
    "null",
    "NaN",
    "nan",
]

# A ordem é definida uma única vez para manter tabelas, gráficos e Power BI consistentes.
FAIXAS_PATRIMONIO = [
    "Sem bens declarados",
    "Até R$ 10 mil",
    "R$ 10 mil a R$ 20 mil",
    "R$ 20 mil a R$ 50 mil",
    "R$ 50 mil a R$ 75 mil",
    "R$ 75 mil a R$ 100 mil",
    "R$ 100 mil a R$ 500 mil",
    "Acima de R$ 500 mil",
]


# ============================================================
# Funções de carregamento e formatação
# ============================================================


def csv_to_dataframe(arquivo: Path) -> pd.DataFrame:
    """Carrega um CSV do TSE em um DataFrame."""
    if not arquivo.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {arquivo}\n"
            "Coloque os CSVs originais dentro da pasta data/raw/."
        )

    # Os arquivos do TSE usam ponto e vírgula, codificação latin-1
    # e podem misturar tipos em colunas muito grandes.
    return pd.read_csv(
        arquivo,
        sep=";",
        encoding="latin-1",
        na_values=VALORES_NULOS,
        low_memory=False,
    )


def converter_valor_brasileiro(valor: object) -> float:
    """Converte valores como '1.500,75' para 1500.75."""
    if pd.isna(valor):
        return np.nan

    if isinstance(valor, (int, float, np.integer, np.floating)):
        return float(valor)

    texto = str(valor).strip().replace(".", "").replace(",", ".")
    return pd.to_numeric(texto, errors="coerce")


def formatar_moeda(valor: float) -> str:
    """Formata um número como moeda brasileira."""
    if pd.isna(valor):
        return "Não disponível"

    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def formatar_numero(valor: float | int) -> str:
    """Formata uma quantidade usando ponto como separador de milhares."""
    return f"{valor:,.0f}".replace(",", ".")


def formatar_percentual(valor: float, casas: int = 2) -> str:
    """Formata um percentual numérico no padrão brasileiro."""
    return f"{valor:.{casas}f}%".replace(".", ",")


# ============================================================
# Preparação dos dados
# ============================================================


def criar_indicador_eleito(vereadores_df: pd.DataFrame) -> pd.DataFrame:
    """Cria uma coluna booleana preservando situações não informadas.

    True  -> situação começa com "ELEITO";
    False -> situação foi informada, mas não começa com "ELEITO";
    <NA>  -> situação eleitoral não informada.
    """
    vereadores_df = vereadores_df.copy()

    situacao = (
        vereadores_df["DS_SIT_TOT_TURNO"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    # Começamos com <NA> para não confundir ausência de resultado com derrota.
    # Depois classificamos apenas as linhas em que a situação eleitoral foi informada.
    eleito = pd.Series(pd.NA, index=vereadores_df.index, dtype="boolean")
    mascara_informada = situacao.notna()
    eleito.loc[mascara_informada] = situacao.loc[mascara_informada].str.match(
        r"^ELEITO\b",
        na=False,
    )

    vereadores_df["eleito"] = eleito
    return vereadores_df


def agregar_patrimonio(bens_df: pd.DataFrame) -> pd.DataFrame:
    """Soma o patrimônio e conta os bens declarados por candidato."""
    bens_df = bens_df.copy()
    # O valor vem como texto no padrão brasileiro, por exemplo "25.000,00".
    # A conversão cria uma coluna numérica apropriada para soma e análise estatística.
    bens_df["VR_BEM_CANDIDATO_NUM"] = bens_df["VR_BEM_CANDIDATO"].apply(
        converter_valor_brasileiro
    )

    # A base de bens possui uma linha por bem. Agrupamos pela chave do candidato
    # para obter uma única linha com patrimônio total e quantidade de bens.
    patrimonio_df = (
        bens_df.groupby("SQ_CANDIDATO", as_index=False)
        .agg(
            patrimonio_total=("VR_BEM_CANDIDATO_NUM", "sum"),
            quantidade_bens=("SQ_CANDIDATO", "size"),
        )
        .copy()
    )

    return patrimonio_df


def calcular_idade_na_eleicao(vereadores_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a idade completa do candidato na data da eleição."""
    vereadores_df = vereadores_df.copy()

    vereadores_df["DT_NASCIMENTO"] = pd.to_datetime(
        vereadores_df["DT_NASCIMENTO"],
        format="%d/%m/%Y",
        errors="coerce",
    )
    vereadores_df["DT_ELEICAO"] = pd.to_datetime(
        vereadores_df["DT_ELEICAO"],
        format="%d/%m/%Y",
        errors="coerce",
    )

    nascimento = vereadores_df["DT_NASCIMENTO"]
    eleicao = vereadores_df["DT_ELEICAO"]

    # Subtrair apenas os anos pode acrescentar um ano indevidamente.
    # Esta máscara identifica quem ainda não havia feito aniversário na data da eleição.
    aniversario_ainda_nao_ocorreu = (
        (eleicao.dt.month < nascimento.dt.month)
        | (
            (eleicao.dt.month == nascimento.dt.month)
            & (eleicao.dt.day < nascimento.dt.day)
        )
    )

    idade = eleicao.dt.year - nascimento.dt.year - aniversario_ainda_nao_ocorreu
    idade = idade.astype("Int64")

    # Idades fora do intervalo plausível são tratadas como ausentes.
    idade = idade.where(idade.between(18, 120))
    vereadores_df["idade"] = idade

    return vereadores_df


def criar_faixas_etarias(vereadores_df: pd.DataFrame) -> pd.DataFrame:
    """Cria faixas etárias interpretáveis e faixas pela regra de Sturges."""
    vereadores_df = vereadores_df.copy()

    # Estas faixas foram escolhidas pela facilidade de interpretação no dashboard.
    vereadores_df["faixa_etaria_fases"] = pd.cut(
        vereadores_df["idade"],
        bins=[17, 25, 30, 35, 40, 50, 60, 80, 100, 120],
        labels=[
            "Até 25 anos",
            "26 a 30 anos",
            "31 a 35 anos",
            "36 a 40 anos",
            "41 a 50 anos",
            "51 a 60 anos",
            "61 a 80 anos",
            "81 a 100 anos",
            "Acima de 100 anos",
        ],
        include_lowest=True,
    )

    idades_validas = vereadores_df["idade"].dropna().astype(int)
    if idades_validas.empty:
        vereadores_df["faixa_etaria"] = pd.NA
        return vereadores_df

    # A regra de Sturges fornece uma segunda divisão, baseada no tamanho da base.
    # Ela é útil para mostrar uma alternativa estatística às faixas interpretativas.
    n = len(idades_validas)
    idade_minima = int(idades_validas.min())
    idade_maxima = int(idades_validas.max())
    amplitude_total = idade_maxima - idade_minima
    numero_classes = max(1, int(np.ceil(1 + 3.322 * np.log10(n))))
    amplitude_classe = max(1, int(np.ceil(amplitude_total / numero_classes)))

    limite_superior = idade_maxima + amplitude_classe
    bins_idade = np.arange(
        idade_minima,
        limite_superior + amplitude_classe,
        amplitude_classe,
    )

    if bins_idade[-1] <= idade_maxima:
        bins_idade = np.append(bins_idade, bins_idade[-1] + amplitude_classe)

    labels = [
        f"{int(inicio)} a {int(fim - 1)} anos"
        for inicio, fim in zip(bins_idade[:-1], bins_idade[1:])
    ]

    vereadores_df["faixa_etaria"] = pd.cut(
        vereadores_df["idade"],
        bins=bins_idade,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    print("\nCálculo estatístico das faixas etárias:")
    print(f"Quantidade de candidatos com idade válida: {formatar_numero(n)}")
    print(f"Menor idade: {idade_minima} anos")
    print(f"Maior idade: {idade_maxima} anos")
    print(f"Amplitude total: {amplitude_total} anos")
    print(f"Número de classes pela regra de Sturges: {numero_classes}")
    print(f"Amplitude de cada classe: {amplitude_classe} anos")

    return vereadores_df


def criar_faixas_patrimonio(vereadores_df: pd.DataFrame) -> pd.DataFrame:
    """Cria faixas ordenadas de patrimônio declarado."""
    vereadores_df = vereadores_df.copy()

    # Primeiro distribuímos os valores positivos nas faixas monetárias.
    faixa = pd.cut(
        vereadores_df["patrimonio_total"],
        bins=[-np.inf, 10_000, 20_000, 50_000, 75_000, 100_000, 500_000, np.inf],
        labels=FAIXAS_PATRIMONIO[1:],
        include_lowest=True,
    ).astype("string")

    # Zero patrimônio pode significar ausência de bens. Usamos a contagem de registros
    # para separar explicitamente quem não declarou nenhum bem.
    faixa.loc[vereadores_df["quantidade_bens"].eq(0)] = "Sem bens declarados"

    vereadores_df["faixa_patrimonio"] = pd.Categorical(
        faixa,
        categories=FAIXAS_PATRIMONIO,
        ordered=True,
    )

    return vereadores_df


# ============================================================
# Estatística descritiva e inferencial
# ============================================================


def resumo_taxa_eleicao(df: pd.DataFrame, coluna_perfil: str) -> pd.DataFrame:
    """Resume candidatos, eleitos, taxa e patrimônio por categoria."""
    # A taxa só pode ser calculada quando o perfil e o resultado eleitoral existem.
    dados = df.dropna(subset=[coluna_perfil, "eleito"]).copy()
    dados["eleito"] = dados["eleito"].astype(bool)

    resumo = (
        dados.groupby(coluna_perfil, observed=True)
        .agg(
            total_candidatos=("SQ_CANDIDATO", "count"),
            total_eleitos=("eleito", "sum"),
            taxa_eleicao=("eleito", "mean"),
            patrimonio_mediano=("patrimonio_total", "median"),
        )
        .reset_index()
    )

    resumo["taxa_eleicao"] *= 100
    return resumo.sort_values("taxa_eleicao", ascending=False)


def teste_associacao_quiquadrado(df: pd.DataFrame, coluna: str) -> dict[str, object]:
    """Testa associação entre uma variável categórica e o resultado eleitoral."""
    dados = df[[coluna, "eleito"]].dropna().copy()
    dados["eleito"] = dados["eleito"].astype(bool)

    # A tabela de contingência cruza as categorias com eleito/não eleito.
    # O teste verifica se as distribuições observadas diferem do esperado sob independência.
    tabela = pd.crosstab(dados[coluna], dados["eleito"])
    tabela = tabela.loc[tabela.sum(axis=1).gt(0), tabela.sum(axis=0).gt(0)]

    if tabela.shape[0] < 2 or tabela.shape[1] < 2:
        return {
            "variavel": coluna,
            "qui_quadrado": np.nan,
            "p_valor": np.nan,
            "graus_liberdade": np.nan,
            "categorias": tabela.shape[0],
        }

    chi2, p_valor, graus_liberdade, _ = chi2_contingency(tabela)

    return {
        "variavel": coluna,
        "qui_quadrado": chi2,
        "p_valor": p_valor,
        "graus_liberdade": graus_liberdade,
        "categorias": tabela.shape[0],
    }


def imprimir_resumo_patrimonio(vereadores_df: pd.DataFrame) -> None:
    """Imprime o describe do patrimônio com nomes acessíveis."""
    nomes_medidas = {
        "count": "Quantidade de candidatos",
        "mean": "Patrimônio médio",
        "std": "Desvio padrão",
        "min": "Menor patrimônio",
        "25%": "25% dos candidatos declararam até",
        "50%": "Mediana: 50% dos candidatos declararam até",
        "75%": "75% dos candidatos declararam até",
        "max": "Maior patrimônio declarado",
    }

    resumo = vereadores_df["patrimonio_total"].describe()
    print("\nResumo do patrimônio declarado:")

    for medida, valor in resumo.items():
        nome = nomes_medidas.get(medida, medida)
        if medida == "count":
            print(f"{nome}: {formatar_numero(valor)}")
        else:
            print(f"{nome}: {formatar_moeda(valor)}")


# ============================================================
# Gráficos
# ============================================================


def escapar_texto_matplotlib(texto: object) -> str:
    """Evita que o símbolo de real seja interpretado como expressão matemática."""
    return str(texto).replace("$", r"\$")


def grafico_taxa_eleicao(
    resumo: pd.DataFrame,
    coluna: str,
    titulo: str,
    nome_arquivo: str,
) -> None:
    """Cria gráfico horizontal da taxa de eleição e mostra os percentuais."""
    dados = resumo.dropna(subset=[coluna, "taxa_eleicao"]).copy()
    dados = dados.sort_values("taxa_eleicao", ascending=True)

    if dados.empty:
        print(f"Gráfico não gerado por ausência de dados: {titulo}")
        return

    categorias = dados[coluna].astype(str).apply(escapar_texto_matplotlib)
    valores = dados["taxa_eleicao"].astype(float)

    # A altura cresce com o número de categorias para evitar rótulos sobrepostos.
    altura = max(5, 0.45 * len(dados) + 2)
    fig, ax = plt.subplots(figsize=(11, altura))
    barras = ax.barh(categorias, valores)

    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel("Taxa de eleição (%)")
    ax.set_ylabel("")

    maior_valor = float(valores.max())
    margem = max(1.0, maior_valor * 0.15)
    ax.set_xlim(0, maior_valor + margem)

    for barra, valor in zip(barras, valores):
        ax.text(
            barra.get_width() + max(0.15, maior_valor * 0.01),
            barra.get_y() + barra.get_height() / 2,
            f"{valor:.1f}%",
            va="center",
            fontsize=10,
        )

    fig.tight_layout()
    caminho = FIGURES_DIR / nome_arquivo
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho}")


# ============================================================
# Modelo preditivo
# ============================================================


def regressao_logistica(
    vereadores_df: pd.DataFrame,
    outputs_dir: Path,
    figures_dir: Path,
) -> tuple[Pipeline, float, pd.DataFrame]:
    """Treina um modelo exploratório para diferenciar eleitos e não eleitos.

    A classe positiva do modelo é "Eleito" (valor 1). Por isso:
    - verdadeiro positivo: candidato eleito previsto corretamente como eleito;
    - verdadeiro negativo: candidato não eleito previsto corretamente como não eleito;
    - falso positivo: candidato não eleito previsto incorretamente como eleito;
    - falso negativo: candidato eleito previsto incorretamente como não eleito.
    """
    print("\n" + "=" * 60)
    print("MODELO PREDITIVO — REGRESSÃO LOGÍSTICA")
    print("=" * 60)

    # O modelo usa somente candidatos cuja situação eleitoral foi informada.
    # Isso evita transformar registros desconhecidos em derrotas.
    dados_modelo = vereadores_df.dropna(subset=["eleito"]).copy()
    dados_modelo["eleito"] = dados_modelo["eleito"].astype(int)

    # O patrimônio tem forte assimetria. A transformação log1p reduz a influência
    # de valores muito altos e continua funcionando quando o patrimônio é zero.
    dados_modelo["log_patrimonio"] = np.log1p(
        dados_modelo["patrimonio_total"]
    )

    variaveis_numericas = [
        "idade",
        "log_patrimonio",
        "quantidade_bens",
    ]

    variaveis_categoricas = [
        "DS_GENERO",
        "DS_GRAU_INSTRUCAO",
        "DS_COR_RACA",
        "DS_OCUPACAO",
        "SG_PARTIDO",
    ]

    # O município não entra no modelo. Além de possuir muitas categorias,
    # ele estava ocupando quase toda a lista de coeficientes e dificultando
    # a interpretação das características individuais dos candidatos.
    variaveis_modelo = variaveis_numericas + variaveis_categoricas

    # O SimpleImputer do scikit-learn espera np.nan nas colunas categóricas.
    for coluna in variaveis_categoricas:
        dados_modelo[coluna] = (
            dados_modelo[coluna]
            .astype("object")
            .where(dados_modelo[coluna].notna(), np.nan)
        )

    X = dados_modelo[variaveis_modelo]
    y = dados_modelo["eleito"]

    if y.nunique() < 2:
        raise ValueError("O modelo precisa de exemplos de eleitos e não eleitos.")

    # Separamos 80% para treinamento e 20% para teste.
    # stratify preserva aproximadamente a proporção de eleitos nos dois grupos.
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    tratamento_numerico = Pipeline(
        steps=[
            ("preencher_nulos", SimpleImputer(strategy="median")),
            ("padronizar", StandardScaler()),
        ]
    )

    tratamento_categorico = Pipeline(
        steps=[
            ("preencher_nulos", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessamento = ColumnTransformer(
        transformers=[
            ("numericas", tratamento_numerico, variaveis_numericas),
            ("categoricas", tratamento_categorico, variaveis_categoricas),
        ]
    )

    # class_weight="balanced" reduz o favorecimento automático da classe
    # majoritária, pois existem muito mais não eleitos do que eleitos.
    modelo = Pipeline(
        steps=[
            ("preprocessamento", preprocessamento),
            (
                "classificador",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2_000,
                    random_state=42,
                ),
            ),
        ]
    )

    modelo.fit(X_treino, y_treino)

    previsoes = modelo.predict(X_teste)
    probabilidades = modelo.predict_proba(X_teste)[:, 1]
    auc_roc = roc_auc_score(y_teste, probabilidades)

    print(f"Candidatos usados no modelo: {formatar_numero(len(dados_modelo))}")
    print(f"Registros de treinamento: {formatar_numero(len(X_treino))}")
    print(f"Registros de teste: {formatar_numero(len(X_teste))}")
    print(f"ROC AUC: {auc_roc:.3f}")

    print("\nRelatório de classificação:")
    print(
        classification_report(
            y_teste,
            previsoes,
            target_names=["Não eleito", "Eleito"],
            zero_division=0,
        )
    )

    # A função confusion_matrix usa a ordem abaixo quando labels=[0, 1]:
    # [[verdadeiro negativo, falso positivo],
    #  [falso negativo, verdadeiro positivo]].
    matriz = confusion_matrix(y_teste, previsoes, labels=[0, 1])
    verdadeiro_negativo, falso_positivo, falso_negativo, verdadeiro_positivo = (
        matriz.ravel()
    )

    print("\nMatriz de confusão — classe positiva: Eleito")
    print("-" * 72)
    print(
        f"Verdadeiro negativo (TN): {verdadeiro_negativo:>5} | "
        "Não eleitos previstos corretamente como não eleitos"
    )
    print(
        f"Falso positivo      (FP): {falso_positivo:>5} | "
        "Não eleitos previstos incorretamente como eleitos"
    )
    print(
        f"Falso negativo      (FN): {falso_negativo:>5} | "
        "Eleitos previstos incorretamente como não eleitos"
    )
    print(
        f"Verdadeiro positivo (TP): {verdadeiro_positivo:>5} | "
        "Eleitos previstos corretamente como eleitos"
    )

    # A tabela 2x2 mantém a estrutura tradicional da matriz de confusão.
    matriz_df = pd.DataFrame(
        [
            [verdadeiro_negativo, falso_positivo],
            [falso_negativo, verdadeiro_positivo],
        ],
        index=["Real: Não eleito", "Real: Eleito"],
        columns=["Previsto: Não eleito", "Previsto: Eleito"],
    )
    matriz_df.to_csv(
        outputs_dir / "matriz_confusao_regressao_logistica.csv",
        sep=";",
        encoding="utf-8-sig",
    )

    # Esta versão em formato longo registra também o nome e o significado
    # de cada célula, facilitando a leitura fora do Python.
    resumo_matriz_df = pd.DataFrame(
        {
            "sigla": ["TN", "FP", "FN", "TP"],
            "tipo": [
                "Verdadeiro negativo",
                "Falso positivo",
                "Falso negativo",
                "Verdadeiro positivo",
            ],
            "valor": [
                verdadeiro_negativo,
                falso_positivo,
                falso_negativo,
                verdadeiro_positivo,
            ],
            "interpretacao": [
                "Não eleitos previstos corretamente como não eleitos",
                "Não eleitos previstos incorretamente como eleitos",
                "Eleitos previstos incorretamente como não eleitos",
                "Eleitos previstos corretamente como eleitos",
            ],
        }
    )
    resumo_matriz_df.to_csv(
        outputs_dir / "resumo_matriz_confusao.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    # Também salvamos uma representação visual da matriz de confusão.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(matriz)
    ax.set_xticks([0, 1], labels=["Previsto: Não eleito", "Previsto: Eleito"])
    ax.set_yticks([0, 1], labels=["Real: Não eleito", "Real: Eleito"])
    ax.set_title("Matriz de confusão — regressão logística")

    nomes_celulas = [
        ["TN", "FP"],
        ["FN", "TP"],
    ]
    descricoes_celulas = [
        ["Verdadeiro negativo", "Falso positivo"],
        ["Falso negativo", "Verdadeiro positivo"],
    ]

    limite_contraste = matriz.max() / 2
    for linha in range(2):
        for coluna in range(2):
            valor = matriz[linha, coluna]
            cor_texto = "white" if valor > limite_contraste else "black"
            ax.text(
                coluna,
                linha,
                f"{nomes_celulas[linha][coluna]}\n{valor}\n"
                f"{descricoes_celulas[linha][coluna]}",
                ha="center",
                va="center",
                color=cor_texto,
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(
        figures_dir / "matriz_confusao_regressao_logistica.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    metricas = classification_report(
        y_teste,
        previsoes,
        target_names=["Não eleito", "Eleito"],
        output_dict=True,
        zero_division=0,
    )
    metricas_df = pd.DataFrame(metricas).transpose()
    metricas_df.loc["modelo", "roc_auc"] = auc_roc
    metricas_df.to_csv(
        outputs_dir / "metricas_regressao_logistica.csv",
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    display = RocCurveDisplay.from_predictions(
        y_teste,
        probabilidades,
        name=f"Regressão logística — AUC = {auc_roc:.3f}",
    )
    display.ax_.set_title("Curva ROC — modelo de eleição")
    display.figure_.tight_layout()
    display.figure_.savefig(
        figures_dir / "curva_roc_regressao_logistica.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(display.figure_)

    # Após o one-hot encoding, uma categoria como "VEREADOR" aparece em um
    # nome técnico como "DS_OCUPACAO_VEREADOR". Esta função converte esses
    # nomes para rótulos intuitivos, por exemplo "Ocupação: VEREADOR".
    rotulos_colunas = {
        "idade": "Idade",
        "log_patrimonio": "Patrimônio declarado (escala logarítmica)",
        "quantidade_bens": "Quantidade de bens declarados",
        "DS_GENERO": "Gênero",
        "DS_GRAU_INSTRUCAO": "Escolaridade",
        "DS_COR_RACA": "Cor/raça",
        "DS_OCUPACAO": "Ocupação",
        "SG_PARTIDO": "Partido",
    }

    def nome_variavel_legivel(nome_tecnico: str) -> str:
        """Converte o nome produzido pelo pipeline em um rótulo compreensível."""
        nome_limpo = (
            nome_tecnico
            .replace("numericas__", "", 1)
            .replace("categoricas__", "", 1)
        )

        if nome_limpo in rotulos_colunas:
            return rotulos_colunas[nome_limpo]

        # Testamos os prefixos mais longos primeiro para separar corretamente
        # o nome da coluna do valor da categoria.
        for coluna in sorted(variaveis_categoricas, key=len, reverse=True):
            prefixo = f"{coluna}_"
            if nome_limpo.startswith(prefixo):
                categoria = nome_limpo[len(prefixo):].strip()
                return f"{rotulos_colunas[coluna]}: {categoria}"

        return nome_limpo

    nomes_variaveis = (
        modelo.named_steps["preprocessamento"].get_feature_names_out()
    )
    coeficientes = modelo.named_steps["classificador"].coef_[0]

    importancia_df = pd.DataFrame(
        {
            "variavel_tecnica": nomes_variaveis,
            "coeficiente": coeficientes,
            "coeficiente_absoluto": np.abs(coeficientes),
        }
    )
    importancia_df["variavel"] = importancia_df["variavel_tecnica"].apply(
        nome_variavel_legivel
    )
    importancia_df["direcao"] = np.where(
        importancia_df["coeficiente"] > 0,
        "Maior probabilidade estimada de eleição",
        "Menor probabilidade estimada de eleição",
    )

    # Os coeficientes são apresentados em dois blocos. Assim, valores negativos
    # não ficam misturados acima de valores positivos apenas por terem grande módulo.
    coeficientes_positivos = importancia_df.loc[
        importancia_df["coeficiente"] > 0
    ].sort_values("coeficiente", ascending=False)

    coeficientes_negativos = importancia_df.loc[
        importancia_df["coeficiente"] < 0
    ].sort_values("coeficiente", ascending=True)

    importancia_ordenada_df = pd.concat(
        [coeficientes_positivos, coeficientes_negativos],
        ignore_index=True,
    )
    importancia_ordenada_df.to_csv(
        outputs_dir / "coeficientes_regressao_logistica.csv",
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    print("\nCaracterísticas associadas a MAIOR probabilidade estimada de eleição:")
    print(
        coeficientes_positivos[["variavel", "coeficiente"]]
        .head(10)
        .to_string(index=False)
    )

    print("\nCaracterísticas associadas a MENOR probabilidade estimada de eleição:")
    print(
        coeficientes_negativos[["variavel", "coeficiente"]]
        .head(10)
        .to_string(index=False)
    )

    # O gráfico reúne os dez coeficientes mais positivos e os dez mais negativos.
    # O sinal mostra a direção da associação preditiva; não representa causalidade.
    top_positivos = coeficientes_positivos.head(10)
    top_negativos = coeficientes_negativos.head(10)
    top_coeficientes = pd.concat(
        [top_negativos, top_positivos],
        ignore_index=True,
    ).sort_values("coeficiente")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(top_coeficientes["variavel"], top_coeficientes["coeficiente"])
    ax.axvline(0, linewidth=1)
    ax.set_title("Características com maior peso na regressão logística")
    ax.set_xlabel(
        "Coeficiente: negativo reduz e positivo aumenta a probabilidade estimada"
    )
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(
        figures_dir / "coeficientes_regressao_logistica.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return modelo, auc_roc, importancia_ordenada_df


# ============================================================
# Resumo executivo
# ============================================================


def gerar_resumo_executivo(
    vereadores_df: pd.DataFrame,
    limite_inferior_ic: float,
    limite_superior_ic: float,
    auc_roc: float,
) -> None:
    """Gera um resumo executivo em Markdown."""
    # O resumo separa o total bruto da base do total usado nas comparações eleitorais.
    resultados_validos_df = vereadores_df.dropna(subset=["eleito"]).copy()
    resultados_validos_df["eleito"] = resultados_validos_df["eleito"].astype(bool)

    mascara_eleitos = resultados_validos_df["eleito"]
    mascara_nao_eleitos = ~mascara_eleitos

    total_candidatos = len(vereadores_df)
    total_resultados_validos = len(resultados_validos_df)
    total_sem_resultado = total_candidatos - total_resultados_validos
    total_eleitos = int(mascara_eleitos.sum())
    taxa_eleicao = mascara_eleitos.mean() * 100

    patrimonio_mediano_eleitos = resultados_validos_df.loc[
        mascara_eleitos,
        "patrimonio_total",
    ].median()
    patrimonio_mediano_nao_eleitos = resultados_validos_df.loc[
        mascara_nao_eleitos,
        "patrimonio_total",
    ].median()

    percentual_sem_bens_eleitos = (
        ~resultados_validos_df.loc[mascara_eleitos, "possui_bens"]
    ).mean() * 100
    percentual_sem_bens_nao_eleitos = (
        ~resultados_validos_df.loc[mascara_nao_eleitos, "possui_bens"]
    ).mean() * 100

    resumo_executivo = dedent(
        f"""
        # Resumo executivo

        A análise avaliou {formatar_numero(total_candidatos)} candidaturas a vereador
        nas eleições municipais de 2024 na Bahia. Deste total,
        {formatar_numero(total_resultados_validos)} possuíam situação eleitoral
        informada e {formatar_numero(total_sem_resultado)} estavam sem resultado
        preenchido na base utilizada.

        ## Principais números

        - Total de eleitos: {formatar_numero(total_eleitos)}
        - Taxa de eleição entre candidaturas com resultado informado: {formatar_percentual(taxa_eleicao)}
        - Intervalo binomial exploratório de 95%: {formatar_percentual(limite_inferior_ic * 100)} a {formatar_percentual(limite_superior_ic * 100)}
        - Patrimônio mediano declarado dos eleitos: {formatar_moeda(patrimonio_mediano_eleitos)}
        - Patrimônio mediano declarado dos não eleitos: {formatar_moeda(patrimonio_mediano_nao_eleitos)}
        - Eleitos sem bens declarados: {formatar_percentual(percentual_sem_bens_eleitos)}
        - Não eleitos sem bens declarados: {formatar_percentual(percentual_sem_bens_nao_eleitos)}
        - ROC AUC da regressão logística exploratória: {auc_roc:.3f}

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
        """
    ).strip()

    caminho = OUTPUTS_DIR / "resumo_executivo.md"
    caminho.write_text(resumo_executivo, encoding="utf-8")
    print(f"\nResumo executivo salvo em: {caminho}")


# ============================================================
# Execução principal
# ============================================================


def main() -> None:
    # 1) Carregamento das duas bases originais do TSE.
    print("Carregando dados do TSE...")
    candidatos_df = csv_to_dataframe(ARQUIVO_CANDIDATOS)
    bens_df = csv_to_dataframe(ARQUIVO_BENS)

    print(
        "Base de candidatos: "
        f"{formatar_numero(candidatos_df.shape[0])} linhas e "
        f"{candidatos_df.shape[1]} colunas."
    )
    print(
        "Base de bens: "
        f"{formatar_numero(bens_df.shape[0])} linhas e "
        f"{bens_df.shape[1]} colunas."
    )

    # 2) O desafio trata apenas do cargo de vereador.
    vereadores_df = candidatos_df.loc[
        candidatos_df["DS_CARGO"].eq("VEREADOR")
    ].copy()
    vereadores_df = criar_indicador_eleito(vereadores_df)

    print(f"\nCandidaturas a vereador: {formatar_numero(len(vereadores_df))}")
    print("Situações eleitorais:")
    print(vereadores_df["DS_SIT_TOT_TURNO"].value_counts(dropna=False))
    print("\nIndicador eleito:")
    print(vereadores_df["eleito"].value_counts(dropna=False))

    # 3) A taxa de eleição exclui somente os registros sem situação informada.
    resultados_validos_df = vereadores_df.dropna(subset=["eleito"]).copy()
    resultados_validos_df["eleito"] = resultados_validos_df["eleito"].astype(bool)

    proporcao_eleitos = resultados_validos_df["eleito"].mean()
    n_candidatos_validos = len(resultados_validos_df)
    erro_padrao = np.sqrt(
        proporcao_eleitos * (1 - proporcao_eleitos) / n_candidatos_validos
    )
    z_score = 1.96
    limite_inferior_ic = max(0.0, proporcao_eleitos - z_score * erro_padrao)
    limite_superior_ic = min(1.0, proporcao_eleitos + z_score * erro_padrao)

    print("\nTaxa de eleição entre candidaturas com resultado informado:")
    print(formatar_percentual(proporcao_eleitos * 100))
    print("Intervalo binomial exploratório de 95%:")
    print(
        f"{formatar_percentual(limite_inferior_ic * 100)} a "
        f"{formatar_percentual(limite_superior_ic * 100)}"
    )

    # 4) Transformamos a base de muitos bens por candidato em uma linha por candidato
    # e depois juntamos o resultado à base principal.
    patrimonio_df = agregar_patrimonio(bens_df)
    vereadores_df = vereadores_df.merge(
        patrimonio_df,
        on="SQ_CANDIDATO",
        how="left",
        validate="one_to_one",
    )
    vereadores_df["patrimonio_total"] = vereadores_df["patrimonio_total"].fillna(0.0)
    vereadores_df["quantidade_bens"] = (
        vereadores_df["quantidade_bens"].fillna(0).astype(int)
    )
    vereadores_df["possui_bens"] = vereadores_df["quantidade_bens"].gt(0)

    imprimir_resumo_patrimonio(vereadores_df)

    # 5) Criação das variáveis derivadas usadas nas análises e no Power BI.
    vereadores_df = calcular_idade_na_eleicao(vereadores_df)
    vereadores_df = criar_faixas_etarias(vereadores_df)
    vereadores_df = criar_faixas_patrimonio(vereadores_df)

    print("\nPerfil por gênero:")
    print(vereadores_df["DS_GENERO"].value_counts(dropna=False))
    print("\nPerfil por escolaridade:")
    print(vereadores_df["DS_GRAU_INSTRUCAO"].value_counts(dropna=False))
    print("\nPerfil por faixa etária:")
    print(vereadores_df["faixa_etaria"].value_counts(dropna=False).sort_index())

    # 6) Tabelas comparativas: cada taxa representa a proporção de eleitos
    # dentro da respectiva categoria, e não a participação da categoria no total.
    resumo_genero = resumo_taxa_eleicao(vereadores_df, "DS_GENERO")
    resumo_faixa_etaria = resumo_taxa_eleicao(vereadores_df, "faixa_etaria")
    resumo_faixa_etaria_fases = resumo_taxa_eleicao(
        vereadores_df,
        "faixa_etaria_fases",
    )
    resumo_escolaridade = resumo_taxa_eleicao(
        vereadores_df,
        "DS_GRAU_INSTRUCAO",
    )
    resumo_partido = resumo_taxa_eleicao(vereadores_df, "SG_PARTIDO")
    resumo_patrimonio = resumo_taxa_eleicao(vereadores_df, "faixa_patrimonio")

    colunas_powerbi = [
        "SQ_CANDIDATO",
        "NM_CANDIDATO",
        "NM_URNA_CANDIDATO",
        "NM_UE",
        "SG_PARTIDO",
        "NM_PARTIDO",
        "DS_CARGO",
        "DS_SIT_TOT_TURNO",
        "eleito",
        "DS_GENERO",
        "DS_GRAU_INSTRUCAO",
        "DS_COR_RACA",
        "DS_ESTADO_CIVIL",
        "DS_OCUPACAO",
        "DT_NASCIMENTO",
        "DT_ELEICAO",
        "idade",
        "faixa_etaria",
        "faixa_etaria_fases",
        "patrimonio_total",
        "faixa_patrimonio",
        "quantidade_bens",
        "possui_bens",
    ]

    # 7) Exporta uma base já limpa e enriquecida para o Power BI.
    vereadores_df[colunas_powerbi].to_csv(
        DATA_DIR / "vereadores_ba_2024_powerbi.csv",
        sep=";",
        index=False,
        decimal=",",
        encoding="utf-8-sig",
    )

    resumos_exportacao = {
        "resumo_por_genero.csv": resumo_genero,
        "resumo_por_faixa_etaria.csv": resumo_faixa_etaria,
        "resumo_por_faixa_etaria_fases.csv": resumo_faixa_etaria_fases,
        "resumo_por_escolaridade.csv": resumo_escolaridade,
        "resumo_por_partido.csv": resumo_partido,
        "resumo_por_faixa_patrimonio.csv": resumo_patrimonio,
    }
    for nome_arquivo, dataframe in resumos_exportacao.items():
        dataframe.to_csv(
            OUTPUTS_DIR / nome_arquivo,
            sep=";",
            index=False,
            decimal=",",
            encoding="utf-8-sig",
        )

    # 8) Gera os gráficos principais solicitados no desafio.
    grafico_taxa_eleicao(
        resumo_patrimonio,
        "faixa_patrimonio",
        "Taxa de eleição por faixa de patrimônio declarado",
        "taxa_eleicao_faixa_patrimonio.png",
    )
    grafico_taxa_eleicao(
        resumo_escolaridade,
        "DS_GRAU_INSTRUCAO",
        "Taxa de eleição por escolaridade",
        "taxa_eleicao_escolaridade.png",
    )
    grafico_taxa_eleicao(
        resumo_faixa_etaria,
        "faixa_etaria",
        "Taxa de eleição por faixa etária",
        "taxa_eleicao_faixa_etaria.png",
    )

    # 9) Testes inferenciais para verificar associação com o resultado eleitoral.
    testes_associacao = [
        teste_associacao_quiquadrado(vereadores_df, "DS_GENERO"),
        teste_associacao_quiquadrado(vereadores_df, "DS_GRAU_INSTRUCAO"),
        teste_associacao_quiquadrado(vereadores_df, "faixa_etaria"),
        teste_associacao_quiquadrado(vereadores_df, "faixa_patrimonio"),
        teste_associacao_quiquadrado(vereadores_df, "SG_PARTIDO"),
    ]
    testes_associacao_df = pd.DataFrame(testes_associacao)
    print("\nTestes de associação com eleição:")
    print(testes_associacao_df.to_string(index=False))
    testes_associacao_df.to_csv(
        OUTPUTS_DIR / "testes_associacao_quiquadrado.csv",
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    # 10) Modelo bônus: regressão logística para previsão exploratória.
    _, auc_roc, _ = regressao_logistica(
        vereadores_df,
        OUTPUTS_DIR,
        FIGURES_DIR,
    )

    # 11) Consolida os indicadores e decisões metodológicas em Markdown.
    gerar_resumo_executivo(
        vereadores_df,
        limite_inferior_ic,
        limite_superior_ic,
        auc_roc,
    )

    print("\nAnálise concluída e arquivos exportados com sucesso.")


if __name__ == "__main__":
    main()
