import spacy
from flask import Flask, render_template, request
from sklearn.externals import joblib


app = Flask(__name__)
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
    return text


@app.route('/')
def index():
    return render_template('base.html')


@app.route('/predict', methods=['GET'])
def predict():
    body = request.get_data()
    header = request.headers
    text = request.args['text']
    text = clean(text)
    filename = 'sg_model.pkl'  # get from https://drive.google.com/file/d/1Cp3FkjwgPweynK1n6qPH2Rj1kEmYfLa0/view?usp=sharing
    m1 = joblib.load(open(filename, 'rb'))
    prediction = m1.predict(text)[0]
    mapping = {0: 'offmychest', 1: 'ADD', 2: 'cripplingalcoholism',
               3: 'leaves', 4: 'MenGetRapedToo', 5: 'rapecounseling',
               6: 'addiction', 7: 'ADHD', 8: 'Advice', 9: 'afterthesilence',
               10: 'Agoraphobia', 11: 'AlAnon', 12: 'alcoholicsanonymous',
               13: 'alcoholism', 14: 'Anger', 15: 'Antipsychiatry',
               16: 'Anxiety', 17: 'Anxietyhelp', 18: 'anxietysuccess'}
    return mapping[prediction]
