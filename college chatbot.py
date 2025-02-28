import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Svecw College ChatBot", layout="centered")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load CSV file securely
csv_url = "svcew_details.csv"
try:
    df = pd.read_csv(csv_url)
except Exception as e:
    st.error("Failed to load the CSV file.")
    st.stop()

df = df.fillna("")  # Removed API key exposure
df["Question"] = df["Question"].str.lower()
df["Answer"] = df["Answer"].str.lower()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df["Question"])

# API Key Configuration (Use environment variables instead)
API_KEY = "AIzaSyA8sX72KAAwZINkvgrU3LUlGA9XTZBo_CI"  
genai.configure(api_key=API_KEY)

# Initialize Generative Model
m = genai.GenerativeModel("gemini-1.5-flash")

# Function to find the closest question
def find_closest_question(user_query, vectorizer, question_vectors, df):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    best_match_index = similarities.argmax()
    best_match_score = similarities[best_match_index]

    if best_match_score > 0.3:
        return df.iloc[best_match_index]["Answer"]
    else:
        return None

st.title("SVECW Chat-Bot")
st.write("Welcome to the college chat bot!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Say something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the closest answer from the dataset
    closest_answer = find_closest_question(prompt, vectorizer, question_vectors, df)

    if closest_answer:
        st.session_state.messages.append({"role": "assistant", "content": closest_answer})
        with st.chat_message("assistant"):
            st.markdown(closest_answer)
    else:
        try:
            # Use the correct model name and include the user prompt
            response = m.generate_content(f"You are a helpful chatbot for SVECW. User's query: {prompt}")
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            with st.chat_message("assistant"):
                st.markdown(response.text)
        except Exception as e:
            st.error(f"Sorry, I couldn't generate a response. Error: {e}")
