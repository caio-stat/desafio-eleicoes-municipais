from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"

ARQUIVO_CANDIDATOS = DATA_RAW_DIR / "consulta_cand_2024_BA.csv"

ARQUIVO_BENS = DATA_RAW_DIR / "bem_candidato_2024_BA.csv"

VALORES_NULOS = ["#NULO", "#NE", "NÃO INFORMADO", "NAO INFORMADO", ""]


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

    print("Situações eleitorais encontradas:")
    print(candidatos_df['DS_SIT_TOT_TURNO'].value_counts(dropna=False))

    



if __name__ == "__main__":
    main()