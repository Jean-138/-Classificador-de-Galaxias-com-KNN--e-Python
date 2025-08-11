import h5py # biblioteca para abrir arquivos no formato .h5
import numpy as np  # Aqui importo o numpy, que é para trabalhar com arrays e números de forma eficiente
from sklearn.model_selection import train_test_split  # Importo para dividir meus dados em treino e teste
from sklearn.neighbors import KNeighborsClassifier  # modelo KNN que vou usar para classificar as galáxias
from sklearn.metrics import confusion_matrix, accuracy_score  # Importo funções para avaliar o modelo
import matplotlib.pyplot as plt  # Importo para criar gráficos e mostrar imagens
import seaborn as sns  # Importo para fazer gráficos bonitos, vou usar para a matriz de confusão


# abro o arquivo galaxy10.h5 e pego as imagens e os rótulos que estão dentro dele
with h5py.File("data/galaxy10.h5", "r") as f:
    imagens = np.array(f["images"])  # Extraio todas as imagens e transformo em array do numpy
    rotulos = np.array(f["ans"])     # Extraio os rótulos as classes das imagens, ans sao os rotulos

    print(imagens.shape)  # imprimr o formato dos dados para conferir quantas imagens e o tamanho delas


# Normalizo os valores dos pixels dividindo por 255, para que fiquem entre 0 e 1
imagens = imagens / 255


# Divido os dados em treino e teste para poder treinar o modelo e depois avaliar se ele aprendeu bem
x_train, x_test, y_train, y_test = train_test_split(
    imagens, rotulos, test_size=0.2, random_state=42)  # 20% dos dadospara teste 80 para treino


# Agora eu preciso transformar as imagens em vetores 1D porque o KNN não entende imagens 3D
x_train = x_train.reshape(x_train.shape[0], -1)  # Achato as imagens de treino, ficando (n_imagens, 14307) 
# -1 em uma das dimensões p o NumPy calcule automaticamente o valor correto.

x_test = x_test.reshape(x_test.shape[0], -1)     # Achato as imagens de teste do mesmo jeito

# Só para garantir que está certo, transformo a primeira imagem de teste de volta para 3D para mostrar depois
imagem1_reshape = x_test[0].reshape(69, 69, 3)


# Crio o modelo KNN, passando que quero que ele olhe os 5 vizinhos mais próximos para classificar
modelo = KNeighborsClassifier(n_neighbors=5)


# Treino o modelo usando as imagens de treino e seus rótulos correspondentes
modelo.fit(x_train, y_train)


# Faço previsões usando as imagens de teste, para ver se o modelo consegue adivinhar a classe certa
y_pred = modelo.predict(x_test)


# Calculo a acurácia, que é a porcentagem de acertos que o modelo teve no teste
acc = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {acc:.4f}")  # Imprimo a acurácia com 4 casas decimais (.:4f)


#  MATRIZ DE CONFUSÃO 

# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 6))  # Defino o tamanho da figura para ficar maior e mais legível
# sns.heatmap(cm, annot=True, cmap="Blues")  
# plt.title("Matriz de Confusão - Galáxias")
# plt.xlabel("Previsão")
# plt.ylabel("Valor Real")
# plt.savefig("matriz_confusao.png")
# plt.show()
# ---------------------------------------------------


# Pego a primeira imagem do conjunto de teste para mostrar
imagem1 = x_test[0]
print(imagem1.shape)  # Imprimo o formato dela 
print(imagem1.size)   # Imprimo o número total de pixels para conferir


# Transformo essa imagem achatada (1D) de volta para o formato 3D original (69, 69, 3) 
# para poder mostrar
imagem1_reshape = imagem1.reshape(69, 69, 3)


plt.clf()  # Limpo a figura atual para evitar sobreposição de gráficos


# Coloco um título explicando qual era o valor real e qual
# foi a previsão do modelo para essa imagem
plt.title(f"Verdadeiro: {y_test[0]} | Previsto: {y_pred[0]}")
plt.axis("off")  # Tiro os eixos, números e linhas 




# Lista com os nomes reais das 10 classes do dataset Galaxy10
nomes_classes = [
    "Disturbed Galaxy",
    "Merging Galaxy",
    "Round Smooth Galaxy",
    "In-between Round Smooth Galaxy",
    "Cigar-shaped Smooth Galaxy",
    "Barred Spiral Galaxy",
    "Unbarred Tight Spiral Galaxy",
    "Unbarred Loose Spiral Galaxy",
    "Edge-on Galaxy without Bulge",
    "Edge-on Galaxy with Bulge"
]


# Crio uma nova figura para exibir as 6 primeiras imagens do teste com seus nomes
plt.figure(figsize=(15, 8))

for i in range(6):
    imagem = x_test[i]
    imagem = imagem.reshape(69, 69, 3)
    plt.subplot(2, 3, i+1)
    plt.imshow(imagem)
    plt.title(f"Verdadeiro: {nomes_classes[y_test[i]]}\nPrevisto: {nomes_classes[y_pred[i]]}")
    plt.axis("off")

# Aqui  tudo junto de uma vez as 6 imagens com rótulos
plt.tight_layout()  # Evita que os textos fiquem sobrepostos
plt.show()
