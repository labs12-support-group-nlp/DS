import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

nlp = spacy.load("en_vectors_web_lg")

def clean(text):
    # split text by white space
    text = text.split()
    text = ' '.join(text)
    # lower case
    text = text.lower()
    # lemmatize
    text = nlp(text)
    text = [token.lemma_ for token in text if len(token.lemma_) > 1]
    return ' '.join(text)

