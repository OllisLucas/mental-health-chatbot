import openai
import pandas as pd
import nltk
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

# Carregar stopwords
stop_words = set(stopwords.words('english'))

# Carregar o modelo treinado
model = joblib.load('model/chatbot_model.pkl')

# Configuração da API da OpenAI
openai.api_key = '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'

# Inicialização do Flask
app = Flask(__name__)

# Carregar perguntas e respostas de saúde mental do arquivo CSV
faq_df = pd.read_csv('data/faq_saude_mental.csv')

llm_model_type = 'openai'
llm_model = 'gpt-4o-mini' 

def gerar_resposta(prompt):
    if llm_model_type == 'openai':
        response = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message['content']
    else:
        raise ValueError("Tipo de modelo não suportado")


# Função para encontrar a resposta mais próxima na FAQ
def encontrar_resposta_na_faq(pergunta):
    # Transforme as perguntas da FAQ e a pergunta do usuário em vetores TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    perguntas_vetorizadas = tfidf_vectorizer.fit_transform(faq_df['pergunta'])
    pergunta_vetor = tfidf_vectorizer.transform([pergunta])

    # Calcule a similaridade entre a pergunta do usuário e cada pergunta da FAQ
    similaridades = cosine_similarity(pergunta_vetor, perguntas_vetorizadas).flatten()
    max_similaridade = max(similaridades)
    
    # Defina um limite de similaridade
    limite_similaridade = 0.5  # Ajuste conforme necessário

    # Se a similaridade máxima estiver acima do limite, retorne a resposta correspondente
    if max_similaridade >= limite_similaridade:
        index = similaridades.argmax()
        return faq_df.iloc[index]['resposta']
    else:
        # Caso contrário, retorne None para acionar a OpenAI
        return None

# Função para salvar feedback dos usuários
def salvar_feedback(pergunta, resposta, correta):
    with open('data/feedback.csv', 'a') as f:
        f.write(f"{pergunta},{resposta},{correta}\n")

# Rota principal (exibe a página inicial)
@app.route("/")
def home():
    return render_template("index.html")

# Rota para obter a resposta do chatbot
@app.route("/get", methods=["POST"])
def chatbot_response():
    prompt = request.form["msg"]

    # Verificar se a pergunta está na base de dados com o modelo treinado
    resposta_faq = encontrar_resposta_na_faq(prompt)

    if resposta_faq:
        resposta = resposta_faq
    else:
        # Geração da resposta do chatbot via GPT
        resposta = gerar_resposta(prompt)

    return render_template("response.html", resposta=resposta)



# Rota para receber feedback do usuário
@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        pergunta = request.form['pergunta']
        resposta = request.form['resposta']
        correta = request.form['correta']

        # Salvar o feedback para posterior treinamento
        salvar_feedback(pergunta, resposta, correta)
        return render_template("feedback_response.html", mensagem="Feedback registrado! Obrigado.")
    return render_template("feedback.html")


# Executa o app
if __name__ == "__main__":
    app.run(port=5001, debug=True)
