# Classificador de Galáxias com KNN e Python 

Este é um projeto de classificação de imagens de galáxias usando aprendizado de máquina com o algoritmo **K-Nearest Neighbors (KNN)**.  
Utilizei o conjunto de dados **Galaxy10** para treinar um modelo que identifica o tipo da galáxia a partir da imagem.

O foco foi aprender machine learning de forma prática, exibindo imagens reais e mostrando se o modelo acertou ou errou a previsão.

---

## Tecnologias utilizadas

- **Python 3**
- **Bibliotecas:**
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - h5py
  - astroNN

---

## Como rodar o projeto

1. **Clone o repositório**
   ```bash
   git clone https://github.com/Jean-138/NOME-DO-REPOSITORIO.git
   ```

2. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

3. **Baixe o dataset Galaxy10**

   O dataset pode ser baixado automaticamente usando a biblioteca `astroNN`.  
   Execute o seguinte código Python para baixar o arquivo `galaxy10.h5`:

   ```python
   from astroNN.datasets import galaxy10
   galaxy10.load_data()
   ```

   Isso salvará o arquivo no cache local do `astroNN`.  
   Mais detalhes estão disponíveis na [documentação oficial do Galaxy10](https://astronn.readthedocs.io/en/latest/galaxy10.html).

4. **Execute o projeto**
   ```bash
   python main.py
   ```

---

## Exemplo de saída

O projeto gera uma matriz de confusão para avaliar o desempenho do modelo.

Além disso, exibe imagens reais das galáxias com o tipo verdadeiro e o tipo previsto pelo modelo.

---

## O que aprendi

- Como usar KNN para classificação de imagens  
- Como lidar com dados em formato `.h5`  
- Como visualizar imagens com matplotlib  
- Como gerar e interpretar uma matriz de confusão  

---

##  Licença 



Este projeto está licenciado sob a licença MIT.
