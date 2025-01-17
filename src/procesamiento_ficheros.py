import pandas as pd
import re
import chardet


def get_encoding(path):
    # Open the file in binary mode and read a portion for detection
    rawdata = open(path, "rb").read(1000)
    # Use chardet to detect the encoding
    result = chardet.detect(rawdata)

    # The detected encoding will be in the 'encoding' key of the result dictionary
    encoding = result["encoding"]

    return encoding.lower()


def cruces_read(path: str) -> pd.DataFrame:
    cruces = pd.read_csv("./data/cruces.csv", encoding="latin-1", sep=";")
    return cruces


def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    columns = [
        "Literal completo del vial tratado",
        "Literal completo del vial que cruza",
        "Clase de la via tratado",
        "Clase de la via que cruza",
        "Particula de la via tratado",
        "Particula de la via que cruza",
        "Nombre de la via tratado",
        "Nombre de la via que cruza",
    ]

    for col in columns:
        df[col] = df[col].str.strip()
    return df


def cruces_as_int(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    columns = [
        "Codigo de vía tratado",
        "Codigo de via que cruza o enlaza",
        "Coordenada X (Guia Urbana) cm (cruce)",
        "Coordenada Y (Guia Urbana) cm (cruce)",
    ]
    for col in columns:
        if df[col].dtype != "int64":
            df[col] = df[col].astype("int")
    return df


def direcciones_read(path: str) -> pd.DataFrame:
    direcciones = pd.read_csv(
        "./data/direcciones.csv", encoding="latin-1", sep=";", low_memory=False
    )
    return direcciones


def direcciones_as_int(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pattern = re.compile(r"0*([+-]?(:?\d+))")

    for col in ["Coordenada X (Guia Urbana) cm", "Coordenada Y (Guia Urbana) cm"]:
        df[col] = df[col].apply(lambda x: pattern.match(x).group(1)).astype("int")
    return df


def literal_split(df: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r"(KM\.|NUM)?(\d+)([A-Za-z]*)?")
    df = df.copy()

    df[["Prefijo de numeracion", "Numero", "Sufijo de numeracion"]] = df[
        "Literal de numeracion"
    ].str.extract(pattern)
    df["Numero"] = df["Numero"].astype("int")
    return df


def process_data(
    path_cruces: str, path_direcciones: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        cruces_as_int(clean_names(cruces_read(path_cruces))),
        literal_split(direcciones_as_int(direcciones_read(path_direcciones))),
    )
