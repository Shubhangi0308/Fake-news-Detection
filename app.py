from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the TF-IDF vectorizer and models
tfidf_vector = pickle.load(open('tfidf_vector.pkl', 'rb'))
pass_agg = pickle.load(open('pass_agg_model.pkl', 'rb'))

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
##nltk.download('stopwords')
##nltk.download('punkt')
##nltk.download('wordnet')
stpwrds = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    # Tokenize the text
    words = nltk.word_tokenize(text)
    # Lemmatize and remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words if word not in stpwrds]
    # Join the words back into a single string
    processed_text = ' '.join(words)
    return processed_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        processed_text = preprocess_text(news_text)
        tfidf_news = tfidf_vector.transform([processed_text])
        prediction_pass_agg = pass_agg.predict(tfidf_news)

        result = {
            'Passive Aggressive': prediction_pass_agg[0]
        }

        return render_template('index.html', result=result, news_text=news_text)

if __name__ == '__main__':
    app.run(debug=True)


##from flask import Flask, render_template, request, jsonify
##import pickle
##from sklearn.feature_extraction.text import TfidfVectorizer
##
##app = Flask(__name__)
##
### Load the TF-IDF vectorizer and models
##tfidf_vector = pickle.load(open('tfidf_vector.pkl', 'rb'))
##pass_agg = pickle.load(open('pass_agg_model.pkl', 'rb'))
####svc = pickle.load(open('svm_model.pkl', 'rb'))
####rfc = pickle.load(open('random_forest_model.pkl', 'rb'))
####lr_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
####dt = pickle.load(open('decision_tree_model.pkl', 'rb'))
##
##@app.route('/')
##def home():
##    return render_template('index.html')
##
##@app.route('/predict', methods=['POST'])
##def predict():
##    if request.method == 'POST':
##        news_text = request.form['news_text']
##        tfidf_news = tfidf_vector.transform([news_text])
##        prediction_pass_agg = pass_agg.predict(tfidf_news)
####        prediction_svc = svc.predict(tfidf_news)
####        prediction_rfc = rfc.predict(tfidf_news)
####        prediction_lr = lr_model.predict(tfidf_news)
####        prediction_dt = dt.predict(tfidf_news)
##
##        # You can choose how to handle multiple predictions
##        # For simplicity, let's return the result of one model
##        result = {
##            'Passive Aggressive': prediction_pass_agg[0],
####            'SVM': prediction_svc[0],
####            'Random Forest': prediction_rfc[0],
####            'Logistic Regression': prediction_lr[0],
####            'Decision Tree': prediction_dt[0]
##        }
##
##        return render_template('index.html', result=result, news_text=news_text)
##
##if __name__ == '__main__':
##    app.run(debug=True)
