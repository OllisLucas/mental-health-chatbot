# Chatbot com Treinamento de Modelo e Interface Web

Um chatbot baseado em **Python**, que utiliza técnicas de **Processamento de Linguagem Natural (PLN)** e **Machine Learning**, com uma interface web para interação. O projeto inclui o treinamento de um modelo com **scikit-learn**, manipulação de dados com **pandas**, e integração com a API da **OpenAI**.

---

## 🚀 Funcionalidades

- **Treinamento do modelo:** Treinamento de um modelo Naive Bayes utilizando TF-IDF para análise de texto.
- **Interface Web:** Interface simples construída com Flask para interação com o chatbot.
- **Integração com OpenAI:** Capacidade de responder utilizando a API da OpenAI.
- **Processamento de Texto:** Remoção de stopwords e análise de similaridade textual com scikit-learn.

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.9+
- **Bibliotecas principais:**
  - `openai`: Para integração com a API da OpenAI.
  - `pandas`: Manipulação e análise de dados.
  - `nltk`: Processamento de linguagem natural.
  - `Flask`: Para criação da interface web.
  - `scikit-learn`: Para treinamento e uso de modelos de machine learning.
  - `joblib`: Para serialização do modelo.

---

## 📂 Estrutura do Projeto

```plaintext
.
├── app.py                 # Arquivo principal da interface Flask
├── train_model.py         # Script para treinamento do modelo
├── requirements.txt       # Dependências do projeto
├── static/                # Arquivos estáticos (CSS, JS, imagens)
├── templates/             # Templates HTML para a interface web
├── model/                 # Modelos treinados e salvos
├── README.md              # Documentação do projeto
└── data/                  # Dados para treinamento/testes
```

## 🖥️ Como Executar o Projeto

**1. Clonar o repositório**

```bash
git clone https://github.com/usuario/chatbot-projeto.git
cd chatbot-projeto
```

**2. Configurar o ambiente**

Certifique-se de ter o Python 3.9+ instalado. Crie um ambiente virtual e ative-o:

```bash
python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/macOS:
source venv/bin/activate
```

**3. Instalar as dependências**

Instale as dependências listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

**4. Baixar os corpora do NLTK**

Execute o seguinte comando para baixar os dados necessários:

```python
import nltk
nltk.download('stopwords')
``` 

**5. Treinar o modelo**

Antes de rodar o chatbot, treine o modelo executando:

```python
python train_model.py
```

Isso salvará o modelo treinado na pasta model/.

**6. Executar o servidor**

Inicie o servidor Flask:

```python
python app.py
```

Acesse o chatbot no navegador em http://localhost:5000.

## 🧪 Exemplos de Uso

- **Entrada de Texto**: O usuário pode enviar mensagens na interface web.

![Campo de Pergunta](/assets/EX01.png)

- **Resposta:** O chatbot responderá com base no modelo treinado ou integrará respostas da API da OpenAI.

![Campo de Resposta](/assets/EX02.png)

## 🤝 Integrantes

Este projeto foi desenvolvido pelos seguintes integrantes:

- **Lucas Macedo de Oliveira**  
  - **RA:** 1272118785  
  - **Função:** Desenvolvimento do backend e treinamento do modelo de ML  

- **Victor Hugo Cordeiro fucilieri**  
  - **RA:** 1272123431  
  - **Função:** Implementação da interface web (Flask) e integração com a API da OpenAI  

- **Rafael Miranda Lima**  
  - **RA:** 1272117553  
  - **Função:** Desenvolvimento do backend e treinamento do modelo de ML  

- **Guilherme Curvelo de Jesus Fraga**  
  - **RA:** 1272116404  
  - **Função:** Manipulação de dados com Pandas e ajuste do modelo de ML  

- **César Rafael Goes Vasconcelos Almeida Souza**  
  - **RA:** 1272118044  
  - **Função:** Manipulação de dados com Pandas e ajuste do modelo de ML  

- **Guilherme Reis Correia**  
  - **RA:** 12722123056  
  - **Função:** Implementação da interface web (Flask) e integração com a API da OpenAI