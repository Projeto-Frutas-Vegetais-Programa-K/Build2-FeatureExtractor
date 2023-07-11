#!/bin/python
import pandas as pd
import numpy as np

#dimensão para as imagens funcionarem no resnet50
RESNET50_IMG_DIM = (224, 224)

class TipoFruta:
    #nome que está no csv
    nomes_csv_frutas = ["abacate", "banana", "caqui", "laranja", "limao", "maca", "mamao", "melancia", "morango", "pessego", "pimentao", "tomate"]
    qual_csv_frutas = ["bom", "ruim"]

    #nome apresentável
    nome_bonito = ["Abacate", "Banana", "Caqui", "Laranja", "Limão", "Maçã", "Mamão", "Melancia", "Morango", "Pêssego", "Pimentão", "Tomate"]
    qual_bonito = ["bom", "boa", "bom", "boa", "bom", "boa", "bom", "boa", "bom", "bom", "bom", "bom"]

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
    """
    Função que lê de fato as imagens e faz a transformação da categoria do tipo

    A idéia é usar a relação índice -> nome relacionado, para que trabalhemos com números e no futuro consigamos saber o 'nome' do tipo  (ex: banana boa) por meio do indice. Isso também facilita o caso reverso, pois podemos usar os indices para criar um equivalente para o one hot encoded. Ex indice 0 -> [1 0 0 ...0], indice 1 -> [0 1 0 0 ....0] (ou seja, o numero indica a posição que marcaremos como 1, partindo da esquerda para a direita por ex). Portanto podemos transformar algo como "Banana boa", "Melão ruim", em algo tratável para uma IA em modelo de classificação (ou seja, números). Olhar a pequena classe implementada acima, TipoFrutas que já tem métodos para isso

    Aqui o objetivo é:
        - Pegar o endereço de cada imagem, transformar em um numpy.ndarray
        - Pegar cada categoria de tipo (" por ex, banana boa, maçã ruim", ...) e transformar em one hot encoded (para ser utilizável na rede, pois temos um problema de classificação)

    Parametros:
        dataframe (pd.DataFrame): Um dataframe 'correto', obtido por meio da função csv_to_df

    Retorna:
        lista_imagens ([np.ndarray]): Lista de np.ndarray equivalente de cada imagem. Cada posição contem um ndarray equivalente a imagem, e cada um deve ter RESNET50_IMG_DIM de dimensões.

        lista_categorias ([np.ndarray]): lista equivalente da categorização dos dados em one hot encoding. Cada elemento é o ndarray equivalente one hot encoded relatio ao tipo da imagem.
    """

    #valores "burros" apenas para passar no teste e facilitar sua escrita
    a = [np.zeros(RESNET50_IMG_DIM), np.ones(RESNET50_IMG_DIM)]
    b = [np.array([0, 1, 0, 0])]
    return (a, b)

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

    #valores "burros" apenas para passar no teste e facilitar sua escrita
    a = [np.zeros(2048)]
    return a

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

    treino = pd.DataFrame()
    teste = pd.DataFrame()
    val = pd.DataFrame()

    return (treino, teste, val)
