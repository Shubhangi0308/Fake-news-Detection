import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
##from sklearn.metrics import accuracy_score, confusion_matrix
##from sklearn.svm import LinearSVC
##from sklearn.ensemble import RandomForestClassifier
##from sklearn.linear_model import LogisticRegression
##from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv('/Users/shubhangitandon/Downloads/Fake News/fake_or_real_news.csv')

X = data['text']
Y = data['label']

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Initialize and fit the TF-IDF vectorizer
tfidf_vector = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vector.fit_transform(X_train)
tfidf_test = tfidf_vector.transform(X_test)

# Initialize and train models
pass_agg = PassiveAggressiveClassifier(max_iter=50)
pass_agg.fit(tfidf_train, Y_train)

##svc = LinearSVC()
##svc.fit(tfidf_train, Y_train)
##
##rfc = RandomForestClassifier()
##rfc.fit(tfidf_train, Y_train)
##
##lr_model = LogisticRegression(solver='liblinear', random_state=0)
##lr_model.fit(tfidf_train, Y_train)
##
##dt = DecisionTreeClassifier()
##dt.fit(tfidf_train, Y_train)

# Save the TF-IDF vectorizer and models
with open('tfidf_vector.pkl', 'wb') as f:
    pickle.dump(tfidf_vector, f)

with open('pass_agg_model.pkl', 'wb') as f:
    pickle.dump(pass_agg, f)

##with open('random_forest_model.pkl', 'wb') as f:
##    pickle.dump(rfc, f)
##
##with open('svm_model.pkl', 'wb') as f:
##    pickle.dump(svc, f)
##
##with open('logistic_regression_model.pkl', 'wb') as f:
##    pickle.dump(lr_model, f)
##
##with open('decision_tree_model.pkl', 'wb') as f:
##    pickle.dump(dt, f)
