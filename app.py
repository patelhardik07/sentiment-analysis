import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
ps = PorterStemmer()
nltk.download('stopwords')
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
tf1 = pickle.load(open("tfidf1.pkl", 'rb'))

app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)
    res={}
    for i in range(len(data['comment'])):
        sent = data['comment'][i+1]
        message=sent
        new_corpus=[]
        sent = re.sub('[^a-zA-Z]',' ',sent)
        sent = sent.lower()
        sent = sent.split()
        sent = [ps.stem(word) for word in sent if not word in stopwords.words('english')]
        sent = ' '.join(sent)
        new_corpus.append(sent)
        tf1_new = TfidfVectorizer(vocabulary = tf1.vocabulary_)
        X_tf1 = tf1_new.fit_transform(new_corpus)
        x_new=X_tf1.toarray()
        prediction = model.predict(x_new)
        res[message]=
        if prediction[0] == 1:
            res[message]='Statement is Positive'
        else:
            res[message]='Statement is Negative '
    return jsonify(res)
if __name__ == "__main__":
    app.run(port = 5000, debug=True)
