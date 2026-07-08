from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"

ARQUIVO_CANDIDATOS = DATA_RAW_DIR / "consulta_cand_2024_BA.csv"

ARQUIVO_BENS = DATA_RAW_DIR / "bem_candidato_2024_BA.csv"

#Isso não é legal. Ficamos pressionados e temos que fazer isso.
#Imagina se chega um novo tipo de nulo que não está na lista? Então, vamos tratar isso de forma mais genérica e ampla.
VALORES_NULOS = ["#NULO", "#NE", "NÃO INFORMADO", "NAO INFORMADO", "", "NULO", "N/A", "NA", "NULL", "null", "NaN", "nan"]


def csv_to_dataframe(arquivo: Path) -> pd.DataFrame:
    """Carrega um arquivo CSV em um DataFrame do Pandas.

    Args:
        arquivo (Path): Caminho para o arquivo CSV.

    Returns:
        pd.DataFrame: DataFrame contendo os dados do CSV.
    """
    return pd.read_csv(
        arquivo,
        sep=";", 
        encoding="latin1", 
        na_values=VALORES_NULOS,
        low_memory=False)


def main():

    print("Carregando dados dos candidatos...")
    candidatos_df = csv_to_dataframe(ARQUIVO_CANDIDATOS)
    bens_df = csv_to_dataframe(ARQUIVO_BENS)

    #base de candidatos
    print("Base de candidatos carregada com sucesso!")
    print(f"Linhas: {candidatos_df.shape[0]}, Colunas: {candidatos_df.shape[1]}")

    #base de bens
    print("Base de bens carregada com sucesso!")
    print(f"Linhas: {bens_df.shape[0]}, Colunas: {bens_df.shape[1]}")

    #primeiras colunas da base de candidatos
    print("Primeiras colunas da base de candidatos:")
    print(candidatos_df.head())

    #primeiras colunas da base de bens
    print("Primeiras colunas da base de bens:")
    print(bens_df.head())

    #primeiras linhas da base de candidatos
    print("Primeiras linhas da base de candidatos:")
    print(candidatos_df.head())

    #primeiras linhas da base de bens
    print("Primeiras linhas da base de bens:")
    print(bens_df.head())

    #colunas da base de candidatos
    print("Colunas da base de candidatos:")
    print(candidatos_df.columns.tolist())
    
    #colunas da base de bens
    print("Colunas da base de bens:")
    print(bens_df.columns.tolist())

    print("Cargos encontrados na base de candidatos:")
    print(candidatos_df['DS_CARGO'].value_counts(dropna=False))

    #Vamos atrás dos nossos vereadores, então vamos filtrar a base de candidatos para o cargo de vereador
    vereadores_df = candidatos_df[candidatos_df['DS_CARGO'] == 'VEREADOR']
    print("Linhas da base de vereadores:")
    print(vereadores_df.shape[0])

    #Situações eleitorais dos vereadores
    print("Situações eleitorais dos vereadores:")
    print(vereadores_df['DS_SIT_TOT_TURNO'].value_counts(dropna=False))

    #Trantando situação eleitoral dos vereadores
    situacao_eleitoral = (
        vereadores_df['DS_SIT_TOT_TURNO']
        .astype("string") #converte pra string pq não somos bobos :)
        .str.upper() #deixa tudo maiúsculo
        .str.strip() #tira espaços em branco
    )

    #listando possiveis situações eleito
    #Ora, o que aconteceria se o TSE colocasse um novo tipo de eleito?
    #Então foi usado regex para pegar qualquer situação que comece com "ELEITO" e não apenas "ELEITO POR QP" ou "ELEITO POR MÉDIA"
    #e um \b para garantir que não pegue "ELEITORA" ou "ELEITORAL" por exemplo
    vereadores_df["eleito"] = situacao_eleitoral.str.match(r"^ELEITO\b", na=False)


    print("\nDistribuição entre eleitos e não eleitos:")
    print(vereadores_df["eleito"].value_counts())

    #Taxa de eleição (percentual de vereadores eleitos)
    print("\nPercentual de vereadores eleitos:")
    taxa_eleicao = vereadores_df["eleito"].mean()
    print(f"{taxa_eleicao:.2%}")










if __name__ == "__main__":
    main()