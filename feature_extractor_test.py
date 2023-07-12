#!/bin/python
import pandas as pd
import pytest, cv2
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


def test_transform_df():
    #dado um dataframe válido
    addr = "Dataset-FV.csv" #está no mesmo diretório
    df = csv_to_df(addr)

    #executa o método
    lista_imagens, lista_categorias = transform_df(df)

    assert isinstance(lista_imagens, list) and all(isinstance(i, np.ndarray) and i.shape[:2] == RESNET50_IMG_DIM for i in lista_imagens
                                                   ), "Primeiro valor de retorno deve ser uma lista de np.ndarrays de tamanho RESNET50_IMG_DIM!"
    assert isinstance(lista_categorias, list) and all(isinstance(i, np.ndarray) for i in lista_categorias), "Segundo valor de retorno deve ser uma lista de np.ndarrays!"

    f = lambda x: np.sum(x)
    assert all(f(k) == 1 for k in lista_categorias), "Todos os elementos devem ser one hot encoded!"

def test_extract_features():
    #dado uma lista de imagens válidas
    lista_imagens = [np.zeros(RESNET50_IMG_DIM)]

    #executa o método
    features = extract_features(lista_imagens)

    assert isinstance(features, list) and all(isinstance(i, np.ndarray) and len(i) == 2048 for i in features
                     ), "Valor de retorno deve ser uma lista de np.ndarrays de 2048 features cada!"



def test_split_into_dataframes():

    #valores burros para testar a função de splitar em
    features = [np.zeros(2048) for i in range(0, 10)]
    lista_categorias = [np.array([0, 1, 0, 0]) for i in range(0, 10)]

    train_split = 0.8
    validation_split = 0.2

    # Metodo interno que checa se o dataframe é one hot encoded
    def is_one_hot_encoded(dataframe):
        for _, row in dataframe.iterrows():
            #garante que tem apenas um único valor não nulo
            if row.astype(bool).sum() != 1:
                return False

            # garante que este valor não nulo é o 1
            if row[row != 0].sum() != 1:
                return False
        return True

    #executa o metodo
    treino, teste, val = split_into_dataframes(features, lista_categorias, train_split, validation_split)

    assert isinstance(treino, pd.DataFrame), "Primeiro retorno deve ser um DataFrame do pandas!"
    assert isinstance(teste, pd.DataFrame), "Segundo retorno deve ser um DataFrame do pandas!"
    assert isinstance(val, pd.DataFrame), "Terceiro retorno deve ser um DataFrame do pandas!"

    for df in [treino, teste, val]:
        #extrai o numero de colunas do dataframe
        n_columns = df.shape[1]

        assert n_columns >= 2048, "Dataframe não contêm as 2048 features esperadas!"
        assert n_columns > 2048, "Dataframe não contêm colunas representando a saida one hot encoded!"

        #caso os asserts passem... extrai as colunas do one hot encoding
        ohe_df = df.iloc[:, 2048:]

        assert is_one_hot_encoded(ohe_df), "Colunas após as features não são do tipo one hot encoded conforme o esperado"
