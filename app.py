import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
import pickle
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
ps = PorterStemmer()
nltk.download('stopwords')
model = pickle.load(open('model.pkl', 'rb'))
tf1 = pickle.load(open("tfidf1.pkl", 'rb'))

app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={
    r"/*": {
        "origins": "*"
    }
})
# routes
@app.route('/', methods=['POST'])
#@crossdomain(origin='*')
def predict():
    
    # get data
    data = request.get_json(force=True)
    res={}
    tf1_new = TfidfVectorizer(vocabulary = tf1.vocabulary_)
    for i in (data['comment']):
        sent = data['comment'][i]
        message=sent
        new_corpus=[]
        sent = re.sub('[^a-zA-Z]',' ',sent)
        sent = sent.lower()
        sent = sent.split()
        sent = [ps.stem(word) for word in sent if not word in stopwords.words('english')]
        sent = ' '.join(sent)
        new_corpus.append(sent)
        X_tf1 = tf1_new.fit_transform(new_corpus)
        x_new=X_tf1.toarray()
        prediction = model.predict(x_new)
        res[i]={}
        if prediction[0] == 1:
            res[i]['comment']=data['comment'][i]
            res[i]['sentiment']="Postive"
        else:
            res[i]['comment']=data['comment'][i]
            res[i]['sentiment']="Negative"
    return jsonify(res)
if __name__ == "__main__":
    app.run(port = 5000, debug=True)
