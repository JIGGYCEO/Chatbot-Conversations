import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity      
import pandas as pd
import warnings
import streamlit as st 
warnings.filterwarnings('ignore')
# import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()
# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

data = pd.read_csv('Conversation 2.csv')
data.drop('Unnamed: 0', axis = 1, inplace = True)
# data

# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    global tokens
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)


data['tokenized question'] = data['question'].apply(preprocess_text)
# data.head(20)

corpus = data['tokenized question'].to_list()
# corpus

tfidf_vector = TfidfVectorizer()
v_corpus = tfidf_vector.fit_transform(corpus)
# print(v_corpus)

st.markdown("<h1 style = 'color: #CE5A67; text-align: center; font-family: Times New Roman, Times, serif;'>Conversations with Tito Chatbot</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #00A9FF; text-align: center; font-family: cursive '>Built By Tito</h4>", unsafe_allow_html = True)

st.markdown("<br> <br>", unsafe_allow_html= True)
col1, col2 = st.columns(2)
col1.image('pngwing.com (14).png', caption = 'Conversations with Tito')

def bot_response(user_input):
    user_input_processed = preprocess_text(user_input)
    v_input = tfidf_vector.transform([user_input_processed])
    most_similar = cosine_similarity(v_input, v_corpus)
    most_similar_index = most_similar.argmax()
    
    return data['answer'].iloc[most_similar_index]

chatbot_greeting = [
    "Hello there, welcome to Conversations with Tito. pls ejoy your usage",
    "Hi user, This bot is created by Tito, Kindly enjoy your usage",
    "Hi hi, How you dey my nigga",
    "Alaye mi, Abeg enjoy your usage",
    "Hey Hey, pls enjoy your usage"    
]

user_greeting = ["hi", "hello there", "hey", "hi there"]
exit_word = ['bye', 'thanks bye', 'exit', 'goodbye']


# # print(f'\t\t\t\t\tWelcome To Conversations with Titos ChatBot\n\n')
# while True:
#     user_q = input('Pls Converse with Tito: ')
#     if user_q in user_greeting:
#         # print(random.choice(chatbot_greeting))
#     elif user_q in exit_word:
#         # print('Thank you for talking to me. Bye')
#         break
#     else:
#         responses = bot_response(user_q)
#         # print(f'ChatBot:  {responses}')

st.write(f'\t\t\t\t\tWelcome To Conversations with Tito ChatBot\n\n')
#while True:
user_q = col2.text_input('Kindly converse with Tito: ')
if user_q in user_greeting:
    col2.write(random.choice(chatbot_greeting))
elif user_q in exit_word:
    col2.write('Thank you for your usage. Bye')
else:
    responses = bot_response(user_q)
    col2.write(f'Tito:  {responses}')