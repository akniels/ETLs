import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier, BallTree
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.neural_network import MLPClassifier
import time
from sklearn.metrics import classification_report
import sys
import pickle



def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    inspector = inspect(engine)
    df = pd.read_sql_table('Message', engine)
    x = df['message']
    
    y = df.drop([ 'id', 'message', 'genre', 'original','categories'], axis=1)
    category_names = ['id', 'message', 'genre', 'original','categories']
    return x,y,category_names

def tokenize(text):
    text = text.lower()
    token = word_tokenize(text)
    words = [w for w in token if w not in stopwords.words('english')]
    stemmed = [PorterStemmer().stem(w) for w in words]
    return token


def build_model():
    pipeline = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(KNeighborsClassifier()))])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for index ,column in enumerate(Y_test, 0):
        f1 = classification_report(Y_test[column].values, [row[index] for row in y_pred])
        print( column)
        print(f1)


def save_model(model, model_filepath):
    pkl_filename = model_filepath
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

# Load from file
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()