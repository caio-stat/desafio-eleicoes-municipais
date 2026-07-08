from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"

ARQUIVO_CANDIDATOS = DATA_RAW_DIR / "consulta_cand_2024_BA.csv"

ARQUIVO_BENS = DATA_RAW_DIR / "bem_candidato_2024_BA.csv"

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


if __name__ == "__main__":
    main()
