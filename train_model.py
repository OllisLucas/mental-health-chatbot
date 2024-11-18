import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Carregar o dataset de perguntas e respostas
faq_df = pd.read_csv('data/faq_saude_mental.csv')

# Dividir os dados em perguntas (X) e respostas (y)
X = faq_df['pergunta']
y = faq_df['resposta']

# Criar um pipeline com vetorização TF-IDF e um classificador Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Treinar o modelo
model.fit(X, y)

# Salvar o modelo treinado
joblib.dump(model, 'model/chatbot_model.pkl')

print("Modelo treinado e salvo com sucesso!")
