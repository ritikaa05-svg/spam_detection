import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the saved model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


# Define the text transformation function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


# Create the web application interface
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input text
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize the preprocessed text
    vector_input = tfidf.transform([transformed_sms])

    # 3. Make a prediction
    result = model.predict(vector_input)[0]

    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")