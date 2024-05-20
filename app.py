import streamlit as st
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from snowballstemmer import TurkishStemmer
from gensim.models import FastText

# Gerekli NLTK verilerini indirin
nltk.download('punkt')
nltk.download('stopwords')

# NLTK Türkçe stop words listesini yükleyin
stop_words = set(stopwords.words('turkish'))

# Streamlit uygulamasını tanımlayın
def main():
    st.title('FastText Modeli Eğitimi ve Sınıflandırma')

    # Dosya yükleme alanı ekleyin
    uploaded_file = st.file_uploader("Lütfen bir xlsx dosyası yükleyin", type=["xlsx"])

    if uploaded_file is not None:
        # Dosyayı okuyun
        df = pd.read_excel(uploaded_file)

        # Metin ön işleme fonksiyonunu uygulayın
        def preprocess_text(text):
            # Tokenization
            tokens = word_tokenize(text)

            # Removing punctuation
            table = str.maketrans('', '', string.punctuation)
            stripped = [word.translate(table) for word in tokens]

            # Removing stopwords
            words = [word for word in stripped if word not in stop_words]

            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]

            # Stemming
            stemmer = TurkishStemmer()
            stemmed_words = [stemmer.stemWord(word) for word in words]

            return ' '.join(stemmed_words)

        df['processed_text'] = df['Haber'].apply(preprocess_text)

        # FastText modelini eğitin
        model = train_fasttext_model(df['processed_text'])

        st.success('FastText modeli başarıyla eğitildi!')

        # Örnek metin giriş alanı ekleme
        st.subheader('Metin Sınıflandırma')
        text_input = st.text_input('Metin Giriniz:', '')

        if text_input:
            # Girilen metni ön işleme yapın
            processed_input = preprocess_text(text_input)

            # Model tarafından tahmin edilen sınıfı bulun
            predicted_class = model.predict([processed_input])[0]

            # Tahmin edilen sınıfı göster
            st.write(predicted_class)

# FastText modeli eğitimi için gerekli fonksiyonu tanımlayın
def train_fasttext_model(data):
    # Veriyi tokenleştirin (bölün)
    tokenized_data = [text.split() for text in data]

    # FastText modelini eğitin
    model = FastText(tokenized_data, size=100, window=5, min_count=1, workers=4, sg=1)

    return model

# Uygulamayı çalıştırın
if __name__ == '__main__':
    main()
