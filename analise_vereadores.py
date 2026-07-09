from pathlib import Path
from scipy.stats import chi2_contingency


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"

ARQUIVO_CANDIDATOS = DATA_RAW_DIR / "consulta_cand_2024_BA.csv"

ARQUIVO_BENS = DATA_RAW_DIR / "bem_candidato_2024_BA.csv"

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures"

OUTPUTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Isso não é legal. Ficamos pressionados e temos que fazer isso.
# Imagina se chega um novo tipo de nulo que não está na lista? Então, vamos tratar isso de forma mais genérica e ampla.
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


def csv_to_dataframe(arquivo: Path) -> pd.DataFrame:
    """Carrega um arquivo CSV em um DataFrame do Pandas.

    Args:
        arquivo (Path): Caminho para o arquivo CSV.

    Returns:
        pd.DataFrame: DataFrame contendo os dados do CSV.
    """
    return pd.read_csv(
        arquivo, sep=";", encoding="latin1", na_values=VALORES_NULOS, low_memory=False
    )


def main():

    print("Carregando dados dos candidatos...")
    candidatos_df = csv_to_dataframe(ARQUIVO_CANDIDATOS)
    bens_df = csv_to_dataframe(ARQUIVO_BENS)

    # base de candidatos
    print("Base de candidatos carregada com sucesso!")
    print(f"Linhas: {candidatos_df.shape[0]}, Colunas: {candidatos_df.shape[1]}")

    # base de bens
    print("Base de bens carregada com sucesso!")
    print(f"Linhas: {bens_df.shape[0]}, Colunas: {bens_df.shape[1]}")

    # primeiras colunas da base de candidatos
    print("Primeiras colunas da base de candidatos:")
    print(candidatos_df.head())

    # primeiras colunas da base de bens
    print("Primeiras colunas da base de bens:")
    print(bens_df.head())

    # primeiras linhas da base de candidatos
    print("Primeiras linhas da base de candidatos:")
    print(candidatos_df.head())

    # primeiras linhas da base de bens
    print("Primeiras linhas da base de bens:")
    print(bens_df.head())

    # colunas da base de candidatos
    print("Colunas da base de candidatos:")
    print(candidatos_df.columns.tolist())

    # colunas da base de bens
    print("Colunas da base de bens:")
    print(bens_df.columns.tolist())

    print("Cargos encontrados na base de candidatos:")
    print(candidatos_df["DS_CARGO"].value_counts(dropna=False))

    # Vamos atrás dos nossos vereadores, então vamos filtrar a base de candidatos para o cargo de vereador
    vereadores_df = candidatos_df[candidatos_df["DS_CARGO"] == "VEREADOR"]
    print("Linhas da base de vereadores:")
    print(vereadores_df.shape[0])

    # Situações eleitorais dos vereadores
    print("Situações eleitorais dos vereadores:")
    print(vereadores_df["DS_SIT_TOT_TURNO"].value_counts(dropna=False))

    # Trantando situação eleitoral dos vereadores
    situacao_eleitoral = (
        vereadores_df["DS_SIT_TOT_TURNO"]
        .astype("string")  # converte pra string pq não somos bobos :)
        .str.upper()  # deixa tudo maiúsculo
        .str.strip()  # tira espaços em branco
    )
    #
    #   Criando a variável "eleito"
    #

    # listando possiveis situações eleito
    # Ora, o que aconteceria se o TSE colocasse um novo tipo de eleito?
    # Então foi usado regex para pegar qualquer situação que comece com "ELEITO" e não apenas "ELEITO POR QP" ou "ELEITO POR MÉDIA"
    # e um \b para garantir que não pegue "ELEITORA" ou "ELEITORAL" por exemplo
    vereadores_df["eleito"] = situacao_eleitoral.str.match(r"^ELEITO\b", na=False)

    print("\nDistribuição entre eleitos e não eleitos:")
    print(vereadores_df["eleito"].value_counts())

    #
    # Taxa de eleição (percentual de vereadores eleitos)
    #
    print("\n Taxa de eleição. Percentual de vereadores eleitos:")
    taxa_eleicao = vereadores_df["eleito"].mean()
    print(f"{taxa_eleicao:.2%}")

    # Intervalo de confiança da taxa de eleição.
    # Intervalo de confiança é um intervalo de valores que tem uma probabilidade específica de conter o valor verdadeiro de uma população.
    # Aqui, estamos calculando o intervalo de confiança de 95% para a taxa de eleição dos vereadores.
    n_candidatos = (
        vereadores_df["eleito"].notna().sum()
    )  # Número de candidatos com informação de eleição
    proporcao_eleitos = vereadores_df["eleito"].mean()  # Proporção de eleitos
    # Distribuição Bernoulli: A distribuição de Bernoulli é uma distribuição de probabilidade discreta
    #  para uma variável aleatória que só pode ter dois resultados possíveis,
    # geralmente representados como 0 e 1. No contexto da eleição,
    # podemos modelar a situação de um candidato como uma variável aleatória de Bernoulli,
    #  onde 1 representa "eleito" e 0 representa "não eleito".
    # A proporção de eleitos (taxa de eleição) é a probabilidade de sucesso (p) na distribuição de Bernoulli.
    # p * (1 - p) / n é a variância da distribuição de Bernoulli, e n é o número de candidatos.
    # Como não é de um único candidato, mas de uma amostra de candidatos,
    # usamos a raiz quadrada da variância para calcular o erro padrão da proporção.
    erro_padrao = np.sqrt(
        proporcao_eleitos * (1 - proporcao_eleitos) / n_candidatos
    )  # Erro padrão da proporção.

    z_score = 1.96  # Z-score para 95% de confiança (consultar tabela Z)

    limite_inferior = proporcao_eleitos - z_score * erro_padrao
    limite_superior = proporcao_eleitos + z_score * erro_padrao

    print("\nIntervalo de confiança de 95% para a taxa de eleição:")
    print(f"Taxa observada: {proporcao_eleitos:.2%}")
    print(f"Limite inferior: {limite_inferior:.2%}")
    print(f"Limite superior: {limite_superior:.2%}")

    # ============================================================

    # Inferência: teste de associação qui-quadrado
    # ============================================================

    def teste_associacao_quiquadrado(df, coluna):
        """
        Testa se existe associação estatística entre uma variável categórica
        e o fato de o candidato ter sido eleito ou não.

        Hipótese nula:
        - Não há associação entre a variável analisada e a eleição.

        Hipótese alternativa:
        - Há associação entre a variável analisada e a eleição.
        """
        dados = df[[coluna, "eleito"]].dropna()

        tabela = pd.crosstab(dados[coluna], dados["eleito"])

        chi2, p_valor, graus_liberdade, _ = chi2_contingency(tabela)

        return {
            "variavel": coluna,
            "qui_quadrado": chi2,
            "p_valor": p_valor,
            "graus_liberdade": graus_liberdade,
            "categorias": tabela.shape[0],
        }

    # Os testes de associação qui-quadrado serão calculados após a criação
    # das variáveis de perfil (`faixa_etaria`, `faixa_patrimonio`).

    def virgula_para_ponto(valor):
        "Converte valores em formato brasileiro para formato americano"
        if pd.isna(valor):
            return 0

        valor = str(valor).strip()  # Remove espaços em branco
        valor = valor.replace(".", "").replace(",", ".")

        try:
            return float(valor)
        except ValueError:
            return 0  # Retorna 0 se a conversão falhar

    bens_df["VR_BEM_CANDIDATO_NUM"] = bens_df["VR_BEM_CANDIDATO"].apply(
        virgula_para_ponto
    )

    # Soma os bens dos candidatos
    patrimonio_df = bens_df.groupby("SQ_CANDIDATO", as_index=False).agg(
        patrimonio_total=("VR_BEM_CANDIDATO_NUM", "sum"),
        quantidade_bens=("VR_BEM_CANDIDATO_NUM", "count"),
    )

    print("\nPatrimônio total e quantidade de bens dos candidatos:")
    print(patrimonio_df.head())
    print(patrimonio_df.shape)

    #
    # Fazendo um Join. Cada um com seus bens
    #
    vereadores_df = vereadores_df.merge(patrimonio_df, on="SQ_CANDIDATO", how="left")

    vereadores_df["patrimonio_total"] = vereadores_df["patrimonio_total"].fillna(0)
    vereadores_df["quantidade_bens"] = (
        vereadores_df["quantidade_bens"].fillna(0).astype(int)
    )

    vereadores_df["possui_bens"] = vereadores_df["quantidade_bens"] > 0

    print("\nVereadores com patrimônio e quantidade de bens:")
    print(
        vereadores_df[
            [
                "SQ_CANDIDATO",
                "NM_CANDIDATO",
                "DS_SIT_TOT_TURNO",
                "eleito",
                "patrimonio_total",
                "quantidade_bens",
                "possui_bens",
            ]
        ].head(15)
    )

    print("\nResumo do patrimonio dos vereadores:")
    print(vereadores_df["patrimonio_total"].describe())

    # Mas os nomes ficaram estranhos, então vamos arrumar isso
    # Deixar mais estatístico e menos técnico, para que qualquer pessoa consiga entender o que significa cada medida
    nomes_medidas = {
        "count": "Quantidade de candidatos",
        "mean": "Patrimônio médio",
        "std": "Desvio padrão",
        "min": "Menor patrimônio",
        "25%": "1º quartil / 25% dos candidatos até",
        "50%": "Mediana / 50% dos candidatos até",
        "75%": "3º quartil / 75% dos candidatos até",
        "max": "Maior patrimônio",
    }

    def formatar_moeda(valor):
        """Formata um valor numérico como moeda brasileira."""
        if pd.isna(valor):
            return "R$ 0,00"
        return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def formatar_numero(valor):
        """
        Formata quantidade no padrão brasileiro.
        Exemplo: 32879 -> 32.879
        """
        return f"{valor:,.0f}".replace(",", ".")

    resumo_patrimonio = vereadores_df["patrimonio_total"].describe()

    for medida, valor in resumo_patrimonio.items():
        nome_amigavel = nomes_medidas.get(medida, medida)

        if medida == "count":
            print(f"\n{nome_amigavel}: {formatar_numero(valor)}")
        else:
            print(f"\n{nome_amigavel}: {formatar_moeda(valor)}")

    # Criando variáveis de perfil

    vereadores_df["DT_NASCIMENTO"] = pd.to_datetime(
        vereadores_df["DT_NASCIMENTO"], format="%d/%m/%Y", errors="coerce"
    )

    vereadores_df["DT_ELEICAO"] = pd.to_datetime(
        vereadores_df["DT_ELEICAO"], format="%d/%m/%Y", errors="coerce"
    )

    vereadores_df["idade"] = (
        vereadores_df["DT_ELEICAO"] - vereadores_df["DT_NASCIMENTO"]
    ).dt.days // 365

    vereadores_df["faixa_etaria_fases"] = pd.cut(
        vereadores_df["idade"],
        # Aqui vale minha experiência de ser humano e não de estatístico.
        # Essas faixas representam não valores quantitativos.
        # São marcos de vida, que representam fases da vida de uma pessoa.
        bins=[0, 25, 30, 35, 40, 50, 60, 80, 100, 120],
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
    )

    # Só que ainda não é estatístico o suficiente.
    # Vamos usar a Regra de Sturges :)
    # Amplitude total = maior valor - menor valor
    # Número de classes = 1 + 3,322 * log10(n) n é o número de observações.
    # Amplitude de cada classe = amplitude total / número de classes

    idades_validas = vereadores_df[
        "idade"
    ].dropna()  # dropna() para evitar problemas com valores nulos
    n = len(idades_validas)
    amplitude_total = idades_validas.max() - idades_validas.min()
    numero_classes = int(
        np.ceil(1 + 3.322 * np.log10(n))
    )  # ceil para arredondar para cima, garantindo que tenhamos classes suficientes

    amplitude_classe = np.ceil(amplitude_total / numero_classes)

    print("\nCálculo estatístico das faixas etárias:")
    print(f"Quantidade de candidatos com idade válida: {n}")
    print(f"Menor idade: {idades_validas.min():.0f} anos")
    print(f"Maior idade: {idades_validas.max():.0f} anos")
    print(f"Amplitude total: {amplitude_total:.0f} anos")
    print(f"Número de classes pela Regra de Sturges: {numero_classes}")
    print(f"Amplitude de cada classe: {amplitude_classe:.0f} anos")

    limite_inferior = idades_validas.min()
    limite_superior = idades_validas.max() + amplitude_classe

    bins_idade = np.arange(
        limite_inferior, limite_superior + amplitude_classe, amplitude_classe
    )

    vereadores_df["faixa_etaria"] = pd.cut(
        vereadores_df["idade"], bins=bins_idade, include_lowest=True, right=True
    )  # pd.cut cria as faixas etárias com base nos limites calculados

    # Deixar os nomes bonitos

    def formatar_intervalo_idade(intervalo):
        if pd.isna(intervalo):
            return "Sem informação"

        inicio = int(np.floor(intervalo.left))
        fim = int(np.floor(intervalo.right))

        return f"{inicio} a {fim} anos"

    vereadores_df["faixa_etaria"] = vereadores_df["faixa_etaria"].apply(
        formatar_intervalo_idade
    )

    print("\nDistribuição/Perfil de candidatos por faixa etária:")
    print(vereadores_df["faixa_etaria"].value_counts(dropna=False).sort_index())

    print("\nDistribuição/Perfil de candidatos por faixa etária (fases da vida):")
    print(vereadores_df["faixa_etaria_fases"].value_counts(dropna=False).sort_index())

    # genero
    print("\nDistribuição/Perfil de candidatos por gênero:")
    print(vereadores_df["DS_GENERO"].value_counts(dropna=False))

    # escolaridade
    print("\nDistribuição/Perfil de candidatos por escolaridade:")
    print(vereadores_df["DS_GRAU_INSTRUCAO"].value_counts(dropna=False))

    #
    # Eleição por perfil
    #

    def resumo_taxa_eleicao(df, coluna_perfil):
        resumo = (
            df.groupby(coluna_perfil)
            .agg(
                total_candidatos=("SQ_CANDIDATO", "count"),
                total_eleitos=("eleito", "sum"),
                taxa_eleicao=("eleito", "mean"),
                # diferença median e mean: mediana valor do meio
                patrimonio_mediano=("patrimonio_total", "median"),
            )
            .reset_index()
        )
        resumo["taxa_eleicao"] = (
            resumo["taxa_eleicao"] * 100
        )  # Convertendo para percentual
        return resumo.sort_values("taxa_eleicao", ascending=False)

    print("\nResumo da taxa de eleição por faixa etária:")
    print(resumo_taxa_eleicao(vereadores_df, "faixa_etaria"))

    print("\nResumo da taxa de eleição por faixa etária (fases da vida):")
    print(resumo_taxa_eleicao(vereadores_df, "faixa_etaria_fases"))

    print("\nResumo da taxa de eleição por gênero:")
    print(resumo_taxa_eleicao(vereadores_df, "DS_GENERO"))

    print("\nResumo da taxa de eleição por escolaridade:")
    print(resumo_taxa_eleicao(vereadores_df, "DS_GRAU_INSTRUCAO"))

    #
    # Faixas de patrimonio
    #
    # esse código ficou ruim pq o matplot não interpretou bem os intervalos de patrimônio, então foi comentado. Mas deixei aqui para referência futura.

    """ vereadores_df["faixa_patrimonio"] = pd.cut(
        vereadores_df["patrimonio_total"],
        bins=[-1, 0, 10000, 20000, 50000, 75000, 100000, 500000, float("inf")],
        labels=[
            "Sem bens declarados",
            "Até R$ 10 mil",
            "R$ 10 mil a R$ 20 mil",
            "R$ 20 mil a R$ 50 mil",
            "R$ 50 mil a R$ 75 mil",
            "R$ 75 mil a R$ 100 mil",
            "R$ 100 mil a R$ 500 mil",
            "Acima de R$ 500 mil",
        ],
    ) """

    # Para gerar esse gráfico de patrimonio tive que apelar muito para IA

    vereadores_df["faixa_patrimonio"] = pd.cut(
        vereadores_df["patrimonio_total"],
        bins=[-1, 0, 10_000, 20_000, 50_000, 75_000, 100_000, 500_000, float("inf")],
        labels=[
            "Sem bens declarados",
            "Até R$ 10 mil",
            "R$ 10 mil a R$ 20 mil",
            "R$ 20 mil a R$ 50 mil",
            "R$ 50 mil a R$ 75 mil",
            "R$ 75 mil a R$ 100 mil",
            "R$ 100 mil a R$ 500 mil",
            "Acima de R$ 500 mil",
        ],
    )

    #
    # Tabelas resumo
    #

    resumo_genero = resumo_taxa_eleicao(vereadores_df, "DS_GENERO")
    resumo_faixa_etaria = resumo_taxa_eleicao(vereadores_df, "faixa_etaria")
    resumo_faixa_etaria_fases = resumo_taxa_eleicao(vereadores_df, "faixa_etaria_fases")
    resumo_escolaridade = resumo_taxa_eleicao(vereadores_df, "DS_GRAU_INSTRUCAO")
    resumo_partido = resumo_taxa_eleicao(vereadores_df, "SG_PARTIDO")
    resumo_patrimonio = resumo_taxa_eleicao(vereadores_df, "faixa_patrimonio")

    print("\nResumo por genero: ")
    print(resumo_genero)

    print("\nResumo por faixa etária: ")
    print(resumo_faixa_etaria)

    print("\nResumo por faixa etária (fases da vida): ")
    print(resumo_faixa_etaria_fases)

    print("\nResumo por escolaridade: ")
    print(resumo_escolaridade)

    print("\nResumo por partido: ")
    print(resumo_partido.head(20))  # Mostrando apenas os 20 primeiros partidos

    print("\nResumo por faixa de patrimônio: ")
    print(resumo_patrimonio)

    #
    # Exportando os dados tratados
    #

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

    vereadores_powerbi_df = vereadores_df[colunas_powerbi].copy()

    vereadores_powerbi_df.to_csv(
        DATA_DIR / "vereadores_ba_2024_powerbi.csv",
        sep=";",
        index=False,
        decimal=",",
        encoding="utf-8-sig",  # utf-8-sig para evitar problemas de acentuação no Power BI
    )

    resumo_genero.to_csv(
        OUTPUTS_DIR / "resumo_por_genero.csv",
        sep=";",
        index=False,
        decimal=",",
        encoding="utf-8-sig",
    )
    resumo_faixa_etaria.to_csv(
        OUTPUTS_DIR / "resumo_por_faixa_etaria.csv",
        sep=";",
        index=False,
        decimal=",",
        encoding="utf-8-sig",
    )
    resumo_faixa_etaria_fases.to_csv(
        OUTPUTS_DIR / "resumo_por_faixa_etaria_fases.csv",
        sep=";",
        index=False,
        decimal=",",
        encoding="utf-8-sig",
    )
    resumo_escolaridade.to_csv(
        OUTPUTS_DIR / "resumo_por_escolaridade.csv",
        sep=";",
        index=False,
        decimal=",",
        encoding="utf-8-sig",
    )
    resumo_partido.to_csv(
        OUTPUTS_DIR / "resumo_por_partido.csv",
        sep=";",
        index=False,
        decimal=",",
        encoding="utf-8-sig",
    )
    resumo_patrimonio.to_csv(
        OUTPUTS_DIR / "resumo_por_faixa_patrimonio.csv",
        sep=";",
        index=False,
        decimal=",",
        encoding="utf-8-sig",
    )

    print("\nDados tratados e resumos exportados com sucesso!")

    def escapar_texto_matplotlib(texto):
        """
        Evita que o matplotlib interprete o símbolo $ como fórmula matemática.
        """
        return str(texto).replace("$", r"\$")

    def grafico_taxa_eleicao(resumo, coluna, titulo, nome_arquivo):
        """
        Cria gráfico de barras horizontais com a taxa de eleição por grupo
        e adiciona o percentual ao lado de cada barra.
        """
        dados = resumo.copy()
        dados = dados.sort_values("taxa_eleicao", ascending=True)

        categorias = dados[coluna].astype(str).apply(escapar_texto_matplotlib)
        valores = dados["taxa_eleicao"]

        fig, ax = plt.subplots(figsize=(11, 6))

        barras = ax.barh(categorias, valores)

        ax.set_title(titulo, fontsize=14)
        ax.set_xlabel("Taxa de eleição (%)")
        ax.set_ylabel("")

        maior_valor = valores.max()
        ax.set_xlim(0, maior_valor * 1.15)

        for barra, valor in zip(barras, valores):
            largura = barra.get_width()
            posicao_y = barra.get_y() + barra.get_height() / 2

            ax.text(
                largura + maior_valor * 0.01,
                posicao_y,
                f"{valor:.1f}%",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()

        caminho = FIGURES_DIR / nome_arquivo
        plt.savefig(caminho, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Gráfico salvo em: {caminho}")

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

    # Voltando ao quiquadrado e exportando ele

    # Agora que `faixa_etaria` e `faixa_patrimonio` foram criadas, rodamos os
    # testes de associação qui-quadrado e construímos o DataFrame de resultados.
    testes_associacao = [
        teste_associacao_quiquadrado(vereadores_df, "DS_GENERO"),
        teste_associacao_quiquadrado(vereadores_df, "DS_GRAU_INSTRUCAO"),
        teste_associacao_quiquadrado(vereadores_df, "faixa_etaria"),
        teste_associacao_quiquadrado(vereadores_df, "faixa_patrimonio"),
        teste_associacao_quiquadrado(vereadores_df, "SG_PARTIDO"),
    ]

    testes_associacao_df = pd.DataFrame(testes_associacao)

    print("\nTestes de associação com eleição:")
    print(testes_associacao_df)

    testes_associacao_df.to_csv(
        OUTPUTS_DIR / "testes_associacao_quiquadrado.csv",
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
