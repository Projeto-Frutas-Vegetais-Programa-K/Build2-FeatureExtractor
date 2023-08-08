#!/bin/python
import pandas as pd
import numpy as np
import re
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model


#base_model = ResNet50(weights='imagenet')
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
#dimensão para as imagens funcionarem no resnet50
RESNET50_IMG_DIM = (224, 224)

class TipoFruta:
    #nome que está no csv
    nomes_csv_frutas = ["abacate", "banana", "caqui", "laranja", "limao", "maca", "mamao", "melancia", "morango", "pessego", "pimentao", "tomate", "pepino"]
    qual_csv_frutas = ["bom", "ruim"]

    #nome apresentável
    nome_bonito = ["Abacate", "Banana", "Caqui", "Laranja", "Limão", "Maçã", "Mamão", "Melancia", "Morango", "Pêssego", "Pimentão", "Tomate", "Pepino"]
    qual_bonito = ["bom", "boa", "bom", "boa", "bom", "boa", "bom", "boa", "bom", "bom", "bom", "bom", "bom"]

    len_nomes = len(nomes_csv_frutas)
    len_qual = len(qual_csv_frutas)


    #retorna o inteiro equivalente
    def get_index(self, nome, qualidade):
        nome_id = self.nomes_csv_frutas.index(nome)
        qual_id = self.qual_csv_frutas.index(qualidade)

        return qual_id*self.len_nomes + nome_id


    #retorna o ndarray one hot encoded a partir do inteiro equivalente
    def one_hot_code_of(self, index):
        tmp = np.zeros(self.len_nomes*self.len_qual)
        tmp[index] = 1
        return tmp

    #retorna o nome "apresentável", com acentos e talz
    def nome_apresentavel(self, index):
        nome = index % self.len_nomes
        return self.nome_bonito[nome]+" "+self.qual_bonito[nome]






def csv_to_df(csv_addr):
    """
    Função que lê o csv do dataset original e retorna um dataframe do pandas.
    O arquivo csv lido deve ser filtrado:
        - Primeira linha  e coluna 'id' são removidos (inuteis).
        - Linhas cuja coluna 'qualidade' estejam marcadas como 'sem_classificacao' devem ser removidas.
    Parametros:
        csv_addr (str): Endereço para o arquivo csv. Ex: "/home/lucas/asdf.csv"
    Retorna:
        ret (pandas.DataFrame): Dataframe do pandas relativo ao csv lido e com filtragem aplicada.
                                Deve começar no indice 0 do dataframe pandas
    """

    # Lê o arquivo csv
    df = pd.read_csv(csv_addr)

    # Remove a primeira linha e a coluna 'id'
    df = df.iloc[1:, 1:]

    # Remove as linhas em que a coluna 'qualidade' está marcada como 'sem_classificacao'
    df = df[df['qualidade'] != 'sem_classificacao']

    # Reinicia o índice do dataframe
    df.reset_index(drop=True, inplace=True)

    return df
    
def transform_df(dataframe):
    tipo_fruta = TipoFruta()
    df = pd.DataFrame(dataframe)
    fp_list = df['arquivo'].to_numpy()
    lista_imagens = []
    lista_categorias = []
    for fp in fp_list:
          splitted = re.split('/', fp)
          if splitted[0] == 'com_classificacao': #caso csv_to_df não tenha sido usado antes
            quality = splitted[1]
            fruit = splitted[2]
            image = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, [224,224])
            lista_imagens.append(image)
            index = tipo_fruta.get_index(fruit, quality)
            lista_categorias.append(index)
    encoded_categorias = list(map(tipo_fruta.one_hot_code_of, lista_categorias))
    return (lista_imagens, encoded_categorias)

def extract_features(lista_imagens):
    """
    Temos aqui a parte central do programa de extração de features.

    A idéia é pegar a lista de imagens obtida com transform_df e passar pela rede resnet50 travada, e gerar o vetor de 2048 caracteristicas.
    Tá praticamente pronto, só revisar o código do estrator de features que ta no 1º notebook do repositório resnet do projeto.

    Parametros:
        lista_imagens ([np.ndarray]): Lista de np.ndarray equivalente de cada imagem.

    Retorna:
        features ([np.ndarray]): Lista de ndarray com as features de cada imagem. Cada imagem será equivalente a um ndarray de 2048 posições (np.array ([f1, f2, ..., f2048]))
    """


     # Carregar o modelo ResNet50 pré-treinado
    base_model = ResNet50(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    # Lista para armazenar as características de cada imagem
    features_array = []

    # Iterar sobre cada imagem na lista
    for imagem in lista_imagens:
        # Redimensionar a imagem para o tamanho esperado pela ResNet50
        imagem = np.resize(imagem, (224, 224, 3))

        # Pré-processar a imagem para a entrada da ResNet50
        imagem = preprocess_input(imagem)

        # Expandir as dimensões da imagem para que seja compatível com a entrada da ResNet50
        imagem = np.expand_dims(imagem, axis=0)

        # Obter as características da imagem usando a ResNet50
        features = model.predict(imagem).reshape(2048,)

        # Adicionar as características à lista de características
        features_array.append(features)

    return features_array

def split_into_dataframes(features, lista_categorias, train_split, validation_split):
    """
    Aqui temos a função que pega as features, e a representação da categoria em one hot encoding e cria 3 dataframes do pandas, um para treino, um para teste e outro para validação

    Para o dataframe do treino, temos o train_split que é o percentual relativo para a parte de treino em cima do total.
    Para o dataframe de teste, temos o restante, ou seja, 1 - train_split de percentual.
    Já para o de validação, calculamos este encima do percentual reservado para o treino (ou seja, para treino mesmo sobra train_split*(1 - validation_split))

    Aqui vale lembrar que deve ser feito um shuffle nas "linhas" do dataframe "mestre", de modo a tentar deixar mais justo antes da divisão. Cada dataframe deve ser "escrito" da seguinte forma:
            - primeiras 2048 colunas devem ser relativas às 2048 features.
            - próximas colunas devem corresponder às colunas do tipo equivalente em one hot encoded


    Parametros:
        features ([np.ndarray]): Lista de ndarray com as features de cada imagem. Cada imagem será equivalente a um ndarray de 2048 posições (np.array ([f1, f2, ..., f2048]))

        lista_categorias ([np.ndarray]): lista equivalente da categorização dos dados em one hot encoding. Cada elemento é o ndarray equivalente one hot encoded relatio ao tipo da imagem.

        train_split (float): percentual reservado para o treino

        validation_split (float): percentual reservado para a validação

    Retorna:
        treino (pd.DataFrame): Um dataframe do pandas representando o dataset de treino.
        teste(pd.DataFrame): Um dataframe do pandas representando o dataset de teste
        val (pd.DataFrame): Um dataframe do pandas representando o dataset de validação


    """

    #função que gera um dataframe "burro" válido, para validar a função de teste
    def generate_dummy_dataframe(nrows):
        # Gerar dados aleatórios para as 2048 colunas
        data = np.random.random((nrows, 2048))

        # Criar o dataframe com as colunas aleatórias
        df = pd.DataFrame(data)

        # Adicionar 4 colunas extras representando a codificação one-hot
        one_hot_columns = pd.DataFrame(np.eye(4)[np.random.choice(4, size=nrows)])
        df = pd.concat([df, one_hot_columns], axis=1)

        return df

    sz_treino_burro = int(len(features)*train_split)
    sz_teste_burro = len(features) - sz_treino_burro

    sz_val_burro = int(sz_treino_burro*validation_split)
    sz_treino_burro = sz_treino_burro - sz_val_burro


    treino = generate_dummy_dataframe(sz_treino_burro)
    teste = generate_dummy_dataframe(sz_teste_burro)
    val = generate_dummy_dataframe(sz_val_burro)

    return (treino, teste, val)

#teste = extract_features(np.zeros([224, 224,3]))
#print(teste)
