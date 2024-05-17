import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
import py_vncorenlp

def vncore(data):
    vncore = py_vncorenlp.VnCoreNLP(save_dir="../Data", annotators=["wseg"], max_heap_size='-Xmx500m')
    data["text"] = data["text"].apply(lambda x: ' '.join(vncore.word_segment(x)))
    return data

def transform_text(text, shortwords_dict):
    # Pattern to match URLs
    text = re.sub(r'http[s]?://\S+', 'link', text)
    
    # Pattern to match 10-digit phone numbers
    text = re.sub(r'\b\d{10}\b', 'phone_number', text)
    
    # Remove duplicated characters at the end of words
    text = re.sub(r'(\w)\1+\b', r'\1', text)
    
    # Replace words exactly with their corresponding replacements
    for word, replacement in shortwords_dict.items():
        text = re.sub(r'\b' + re.escape(word) + r'\b', replacement, text)   
    return text

def preprocess(data):
    # Process data
    data = vncore(data)
    shortwords_dict = pd.read_csv("../Data/vietnamese_shortwords.csv", sep=';', header=None, index_col=0).squeeze().to_dict()
    data['text'] = data['text'].str.lower().apply(transform_text, args=(shortwords_dict,))
    return data

# Load data
data = pd.read_csv("../Data/vietnamese_data.csv").dropna()

# Preprocess data
processed_data = preprocess(data)

# Load stopwords
stopwords = pd.read_csv("../Data/vietnamese_stopwords.csv", header=None).squeeze().tolist()

# Vectorize text
vectorizer = CountVectorizer(stop_words=stopwords)
data_vectorized = vectorizer.fit_transform(processed_data["text"])

# Save processed data, vectorizer, and preprocessing function
with open('../Model/processed_data.pkl', 'wb') as f:
    pickle.dump((processed_data, data_vectorized, vectorizer), f)

print("Data, vectorizer, and preprocessing function saved to processed_data.pkl")
