#!/bin/python
import pandas as pd
import pytest
from feature_extractor import *

# Para funcionar, precisa ter o pytest instalado (pip install pytest)
# para executar basta digitar pytest dentro do diretorio que ele já vascula e detecta
# todas as funções com test_ no inicio


def test_csv_to_df():
    addr = "Dataset-FV.csv" #está no mesmo diretório
    df = csv_to_df(addr)

    assert isinstance(df, pd.DataFrame), "Deve retornar um DataFrame do pandas!"
    assert df.index.min() == 0, "Indice do DataFrame deve começar em 0!"
    assert "id" not in df.columns, "Coluna 'id' deveria ser removida!"
    assert not any(df["qualidade"] == "sem_classificacao"), "As imagens do tipo 'sem_classificacao' devem ser removidas!"
    assert "tipo" in df.columns, "Coluna de 'tipo' deve estar presente no DataFrame!"


@pytest.fixture
def test_transform_df():
    #dado um dataframe válido
    addr = "Dataset-FV.csv" #está no mesmo diretório
    df = csv_to_df(addr)

    #executa o método
    lista_imagens, lista_categorias = transform_df(df)

    assert isinstance(lista_imagens, list) and all(isinstance(i, np.ndarray) for i in lista_imagens), "Primeiro valor de retorno deve ser uma lista de np.ndarrays!"
    assert isinstance(lista_categorias, list) and all(isinstance(i, np.ndarray) for i in lista_categorias), "Segundo valor de retorno deve ser uma lista de np.ndarrays!"

    f = lambda x: np.sum(x)
    assert all(f(k) == 1 for k in lista_categorias), "Todos os elementos devem ser one hot encoded!"

    return lista_imagens

def test_extract_features(test_transform_df):
    #dado uma lista de imagens válidas
    lista_imagens = test_transform_df

    #executa o método
    features = extract_features(lista_imagens)

    assert isinstance(features, list) and all(isinstance(i, np.ndarray) and len(i) == 2048 for i in features
                     ), "Valor de retorno deve ser uma lista de np.ndarrays de 2048 features cada!"


