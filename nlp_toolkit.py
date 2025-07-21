import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
import pandas as pd
import io
from googletrans import Translator
from gtts import gTTS

import os
nltk.download('punkt')


# Download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')
stop_words_nltk = set(stopwords.words('english'))



st.set_page_config(
    page_title = "Home Page",
    page_icon="🏠",
    layout="centered"
    )
def Welcome():
    st.title("Hello There!")

    if "name" not in st.session_state:
        st.session_state["name"] = ""
    st.markdown("Lets begin with introducing yourself")
    name = st.text_input("Every great story starts with a name...whats yours?")

    submit = st.button("Submit")

    if submit:
        if name.strip() != "":
            st.session_state["name"] = name
            st.success(f"Welcome, {name}! Lets start with the NLP App.")

            st.markdown("""
            ### 🚀 How to Get Started:

            - Use the **left sidebar** to choose any NLP tool.
            - Enter or paste your text in the input box.
            - View the results instantly — no coding required!

            Each tool is interactive and gives you real-time output.
            """)
            st.markdown("""
                ---

            ### 🙋 About This App

            This NLP Toolkit is built for students, developers, data scientists, journalists — anyone who works with text data.

            NLP is widely used in applications like chatbots, customer feedback analysis, translation services, and more.

            🛠️ Built using **Python, Streamlit**, and top NLP libraries like **spaCy, Hugging Face Transformers, and NLTK**.

            """)

        else:
         st.error("Please enter your name")
def about():
       
    st.title("🔤 NLP Tools Dashboard")

    st.markdown("""
        ## 👋 Welcome to the NLP Tools App

    This application provides a collection of powerful **Natural Language Processing (NLP)** tools to help you analyze, understand, and transform text data with ease.

    Our goal is to make NLP **simple, interactive, and accessible** for everyone — from beginners to professionals.
    """)
    st.markdown("""
    ### 🧰 What You Can Do:

    1. ** 📝 Text Summarization**  
    Turn long texts into short, meaningful summaries.

    2. **😊Sentiment Analysis**  
    Detect the emotion behind the text — Positive, Negative, or Neutral.

    3. **🧹Text Cleaning App**  
    Clean your text data with ease by removing unnecessary elements such as:
   - Punctuation  
   - Numbers  
   - Stopwords  
   - URLs  
   - And converting to lowercase  
     This helps prepare your text for further analysis or modeling.

    4. **🛠️Text Preprocessing App**
    Explore powerful preprocessing techniques including:
   - **Tokenization** (Split text into words or sentences)  
   - **Stemming** (Reduce words to their base/root form)  
   - **Lemmatization** (Normalize words using vocabulary and grammar)  
   - **POS Tagging** (Part-of-speech tagging for each word)  
                
    5. ** 🗂️ Text Classification**  
    Categorize text into predefined classes, like spam detection or topic sorting.

    6. **🌐Translation**  
    Translate text between different languages.
    7. **🔊Text to Speech App**
    Convert written text into spoken words using AI-powered voices.
    """) 
   
st.sidebar.title("🧠 NLP Toolkit Menu")

option = st.sidebar.selectbox("What would you like to do?",
    [
        'Welcome Page',
        'About NLP App',
        'Text Cleaning App',
        'Text Pre processing Tool',
        'Text Summarization',
        'Text Translation',
        'Text Classification',
        'Sentiment Analysis',
        'Text to Speech App'
    ]
)
## Text Cleaning functions ###
def to_lowercase(text):
        return text.lower()

def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_stopwords(text):
    words = text.split()
    filtered = [word for word in words if word.lower() not in stop_words_nltk]
    return ' '.join(filtered)

def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

### Text Preprocessing Funtions

def pos_tag(text):
    blob = TextBlob(text)
    return (blob.tags)

def word_token(text):
    blob = TextBlob(text)
    return(blob.words)

def sent_token(text):
    blob = TextBlob(text)
    return(blob.sentences)
def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_token(text)
    return [lemmatizer.lemmatize(w) for w in tokens]

def stem_words(text):
    ps = PorterStemmer()
    tokens = word_token(text)
    return [ps.stem(w) for w in tokens]


##Sentiment Analysis Function"

def sent(text1: str):
    data = TextBlob(text1)
    sentiment = data.sentiment
    return sentiment
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    filename = "output.mp3"
    tts.save(filename)
    return filename  # Clean up after playing
if option =="Welcome Page":
    Welcome()

elif option == "About NLP App":
    about()
elif option == "Text Cleaning App":
# --- Streamlit UI ---
    st.title(" 🧼  NLP Text Cleaning App ")
    st.markdown("Clean your messy text to prepare it for analysis.")

# --- File Upload ---
    uploaded_file = st.file_uploader("📁 Upload a text file", type=["txt"])
    if uploaded_file is not None:
        raw_text = uploaded_file.read().decode("utf-8")
    else:
        raw_text = st.text_area("✍️ Paste your text here:", height=250)

# --- Options ---
    if raw_text:
        st.subheader("🔧 Text Cleaning Options")

        col1, col2 = st.columns(2)
        with col1:
            lowercase = st.checkbox("Convert to lowercase", value=True)
            remove_punct = st.checkbox("Remove punctuation")
            remove_nums = st.checkbox("Remove numbers")
        with col2:
            remove_stops = st.checkbox("Remove stopwords")
            remove_urls = st.checkbox("Remove URLs")
        
    # --- Text Cleaning Pipeline ---
        text = raw_text
        cleaned = text
        if st.button("🧹 Clean Text"):

            if not (lowercase or remove_punct or remove_nums or remove_stops or remove_urls):
                st.warning("⚠️ Please select at least one cleaning option.")
            else:
                text = raw_text
                cleaned = text
                if lowercase:
                    cleaned = to_lowercase(cleaned)
                if remove_punct:
                    cleaned = remove_punctuation(cleaned)
                if remove_nums:
                    cleaned = remove_numbers(cleaned)
                if remove_stops:
                    cleaned = remove_stopwords(cleaned)
                if remove_urls:
                    cleaned = remove_url(cleaned)
                st.session_state['cleaned_text'] = cleaned

    # --- Output ---
        if 'cleaned_text' in st.session_state:
            st.subheader("🔍 Cleaned Text")
            st.text_area("Output:", cleaned, height=200)
            st.download_button("📥 Download Cleaned Text", st.session_state['cleaned_text'], file_name="cleaned_text.txt")

elif option == "Text Pre processing Tool":

    st.header("🧹 NLP PrepHub: Text Preprocessing Made Easy")
    st.markdown("Welcome to **NLP PrepHub**, your all-in-one solution for transforming raw text into structured, analysis-ready data.")

    text = st.text_area("Enter/Paste your Text Here", height=100) 
    tool = st.selectbox("Select an NLP Tool", ["Word Tokenization", "Sentence Tokenization", "Stemming", "Lemmatization", "POS Tagging"])
    if st.button("Process"):
        if not text.strip():
            st.error("Please enter some text to process")                   
        else:
            if tool == "POS Tagging":
                st.markdown("**POS Tags:**")
                results = pos_tag(text)
                for word, tag in results:
                    st.success(f"{word}: {tag}")

            elif tool == "Word Tokenization":
                st.markdown("**Word Tokens:**")
                st.success("Result")
                st.write(list(word_token(text)))

            elif tool == "Sentence Tokenization":
                st.markdown("**Sentences:**")
                for sent in sent_token(text):
                    st.write(str(sent))
            elif tool == "Stemming":
                stemmed_words = stem_words(text)
                st.markdown("**Stemmed Words:**")
                st.write(stemmed_words)
            elif tool == "Lemmatization":
                lemmatizer = WordNetLemmatizer()
                tokens = word_token(text)
                lemmas = [lemmatizer.lemmatize(w) for w in tokens]
                st.markdown("**Lemmatized Words:**")
                st.write(lemmas)

   
elif option == "Text Summarization":
        st.header("📝QuickSum")
        st.subheader("⚡ From Long Text to Smart Summary — Instantly!")
        st.markdown("Welcome to **Summarizer App**, your go-to app for converting long articles, documents, or notes into short, meaningful summaries — all in just one click.")

        
        sum_option = st.selectbox(
        "🧠 Please choose a summarization method:",
            ["",
            "🧪 LSA Summarizer",
            "🧾 LexRank Summarizer",
            "📚 Luhn Summarizer"
            ]
        )
           # Upload or enter text
        upload_file = st.file_uploader("Upload a Text file", type=["txt"])

        if upload_file is not None:
            original_text = upload_file.read().decode("utf-8")
        else:
            original_text = st.text_area("Enter Your Text Here", height=150)

        if st.button("Summarize"):
            if original_text.strip() =="":
               st.warning("Please enter text or upload some text file to summarize.")
            else:

                if sum_option == "🧪 LSA Summarizer":
                    parser = PlaintextParser.from_string(original_text, Tokenizer("english"))

                    lsa_summarizer = LsaSummarizer()

                    summarized_text = lsa_summarizer(parser.document, sentences_count=10)
                    summary = ' '.join(str(sentence) for sentence in summarized_text)

                    st.success("\nSummary:")
                    st.write(summary)
                    st.download_button("Download Result", data=summary, file_name="summarized_text.txt")


                elif sum_option =="🧾 LexRank Summarizer":
                    parser = PlaintextParser.from_string(original_text, Tokenizer("english"))
                    lex_summarizer1 = LexRankSummarizer()
                    summarized_text2 = lex_summarizer1(parser.document, 2)
                    summary2 = ' '.join(str(sentence) for sentence in summarized_text2)
                    
                    st.success("\nSummary:")
                    st.write(summary2)
                    st.download_button("Download Result", data=summary2, file_name="summarized_text.txt")

                elif sum_option == "📚 Luhn Summarizer":
                    summarizer_luhn = LuhnSummarizer()
                    parser = PlaintextParser.from_string(original_text, Tokenizer("english"))
                    summarized_text3 = summarizer_luhn(parser.document, 3)
                    summary3 = ' '.join(str(sentence) for sentence in summarized_text3)
                    st.success("\nSummary:")
                    st.write(summary3)
                    st.download_button("Download Result", data=summary3, file_name="summarized_text.txt")
                    

elif option == "Sentiment Analysis":
    st.header("Emotion Scope: Explore the Tone of Language")
    st.markdown(" 🧠 Sentiment Analysis App")

    text1 = st.text_area("🔍 Enter Text to Analyze Emotion", height=100)

    if st.button("🔍Analyze"):
        if text1.strip()== "":
            st.warning("Enter text to Analyze")
        else:

            sentiment = sent(text1)
            polarity = sentiment.polarity
            subjectivity = sentiment.subjectivity
            st.write("Sentiment Analysis Result")
            st.write(f"🧭Polarity is {polarity:.3f}")
            st.write(f"🧪Subjectivity is {subjectivity:.3f}") 
                    

            if polarity > 0 :
                st.success("Your Text is Positive😊")
            elif polarity < 0:
                st.error("Your Text is Negative 😠")
            else:
                st.info("Your text is Neutral😐")
            
            if subjectivity < 0.3:
                st.info("🧊Your content is Objective 📰")
            
            elif 0.3 <= subjectivity < 0.6:
                st.info("Your Content is Balanced⚖️")
            else:
                st.info("Your content is Subjective🎨")
        
            data = {
            "Text": [text1],
            "Polarity": [polarity],
            "Subjectivity": [subjectivity]
                }
            df = pd.DataFrame(data)

            # Create CSV and download button
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                        label="📥 Download Result as CSV",
                        data=csv_data,
                        file_name="sentiment_result.csv",
                        mime="text/csv"
                    )
elif option == "Text Translation":
    translator = Translator()
    st.title("LingoBridge – Your Gateway to Global Conversations 🌍")
    st.markdown("Break language barriers effortlessly.")
    text_input = st.text_area("Enter the text to Translate to : ", "")

    languages = {
        "French" : "fr",
        "Spanish" : "es",
        "German" : "de",
        "Chinese (Simplified)" : "zh-CN",
        "Hindi" : "hi",
        "Arabic" : "ar",
        "Russian" : "ru",
        "Japanese": "ja",
        "Urdu" : "ur",
        "Portugues" : "pt",
        "Bengali" : "bn",
        "Danish" : "da",
        "Korean" : "ko",
        "Nepali": "ne",
        "Punjabi(Gurumukhi India)" : "pa"
    }   

    target_lang = st.selectbox("Choose Language to Translate to:", list(languages.keys()))

    if st.button("Translate"):
        if text_input.strip() == "":
            st.warning("Please enter text to translate.")
        else:
            lang_code = languages[target_lang]
            translated = translator.translate(text_input, dest=lang_code)
            st.success(f"Translated Text ({target_lang}):")
            st.write(translated.text)

            tts = gTTS(translated.text, lang=lang_code)
            tts.save("translated_audio.mp3")
            st.audio("translated_audio.mp3", format="audio/mp3")
elif option =="Text Classification":

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    st.title("🧠📚 TextSortAI – Classify Your Text Smarter ✨")

    text = st.text_area("Enter your text to classify", "")
    category_labels = ["Finance", "Politics", "Sports", "Entertainment","Technology", "Others"]

    if st.button("Classify"):
        if text.strip() == "":
            st.warning("Please enter text to classify")
        else: 
            classified_text = classifier(text, category_labels)
            st.success("Classification:")
            st.json(classified_text)

elif option == "Text to Speech App":
    st.header("🔊📄 VocalText – Text-to-Speech Simplified 🎙️")
    st.markdown("Let your text talk, powered by AI.")
    input_text = st.text_area("Enter your text to speak")

    if st.button("Speak"):
        if input_text.strip() == "":
            st.warning("Please enter some text to speak")
        else:
            file = text_to_speech(input_text)
            st.audio(file, format='audio/mp3')
    
# Add 'About Me' section to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 About Me")
st.sidebar.info(
    """
    **Developed by:**
      Maria Hameed  
    📧 Email: hameedmaria06@gmail.com 

    🐍 Passionate about NLP & AI.
    """
)
