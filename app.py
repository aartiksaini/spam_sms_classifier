import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download the required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Initialize PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric tokens
    y = [i for i in text if i.isalnum()]
    
    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    
    # Stem the words
    y = [ps.stem(i) for i in y]
    
    # Return the transformed text as a single string
    return " ".join(y)

# Load the TF-IDF vectorizer and model
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
model = pickle.load(open('knc_model.pkl', 'rb'))

# Streamlit app
st.set_page_config(page_title="Spam Classifier", page_icon="📱", layout="centered")

# Title with emoji
st.title("📩 Email/SMS Spam Classifier 🚫")

st.markdown("""
    <style>
        .title {
            font-size: 30px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="title">Detect if a message is Spam or Not!</p>', unsafe_allow_html=True)

input_sms = st.text_area("✍️ Enter the message here:", "Type your message...")

# Prediction button with emoji
if st.button('🔍 Predict'):
    # 1. Preprocess the input text
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize the preprocessed text
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Predict using the model
    result = model.predict(vector_input)[0]
    
    # 4. Display the result with a better style and emojis
    if result == 1:
        st.header("🚨 **Spam** Message Detected! 🚨")
        st.write("This message is likely a **Spam**. Be cautious before interacting with it.")
        st.image("https://media.giphy.com/media/26AHONQkbfhNKF2D6/giphy.gif", width=400)  # Spam alert gif
    else:
        st.header("✅ **Not Spam**! ✅")
        st.write("This message seems **safe**. No need to worry!")
        st.image("https://media.giphy.com/media/l0MYyQjw2nZYCT4tC/giphy.gif", width=400)  # Safe message gif
