import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text

    y = []
    for i in text:
        if i.isalnum():  # Keep only alphanumeric words
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        # Remove stopwords and punctuation
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        # Apply stemming
        y.append(ps.stem(i))

    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI setup
st.set_page_config(page_title="Email/SMS Spam Classifier", page_icon="📧", layout="wide")

# Custom CSS for more attractive UI
st.markdown("""
    <style>
        body {
            background-color: #f0f0f5;
            font-family: 'Arial', sans-serif;
        }
        .stTextArea textarea {
            background-color: #ffffff;
            color: #333333;  /* Dark text color for readability */
            border-radius: 10px;
            border: 1px solid #dcdcdc;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 12px;
            font-size: 16px;
            transition: 0.3s;
            height: 120px; /* Smaller height for the text box */
        }
        .stTextArea textarea:focus {
            outline: none;
            border-color: #4CAF50;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 12px 24px;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        }
        .result {
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .result-spam {
            background-color: #f8d7da;
            color: #721c24;
        }
        .result-not-spam {
            background-color: #d4edda;
            color: #155724;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            font-size: 14px;
            color: #888888;
        }
        .header-text {
            text-align: center;
            color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description with icons
st.title("📧 **Email/SMS Spam Classifier**")
st.markdown("""
Welcome to the **Spam Classifier**! Enter your message below, and click **Predict** to check whether it is **Spam** or **Not Spam**.
""")

# Input field with placeholder and smaller size
input_sms = st.text_area(
    "Enter the message:",
    placeholder="Type or paste the message here...",
    height=120,  # Reduced height for smaller box
    max_chars=1000
)

# Add a loading spinner when prediction is in progress
if st.button('🔍 **Predict**'):
    with st.spinner('Classifying... Please wait'):
        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize the transformed text
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict the result using the model
        result = model.predict(vector_input)[0]
        
        # 4. Display the result with enhanced visuals and animations
        if result == 1:
            st.markdown("<div class='result result-spam'>🚨 **Spam**</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result result-not-spam'>✅ **Not Spam**</div>", unsafe_allow_html=True)

# Footer with a personalized message
st.markdown("""
<div class="footer">
     Built with Streamlit, NLTK, and Machine Learning
</div>
""", unsafe_allow_html=True)
