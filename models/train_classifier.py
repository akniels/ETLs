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
from nltk.stem import WordNetLemmatizer



def load_data(database_filepath):
    """"
    Loads the data from the database and returns x y, and category names
    
    input = database filepath
    
    output: 
        x = x variable
        y = y variable
        category_names = names of categories for the y variables
    
    
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    inspector = inspect(engine)
    df = pd.read_sql_table('Message', engine)
    x = df['message']
    
    y = df.drop([ 'id', 'message', 'genre', 'original', 'categories'], axis=1)
    category_names = list(y.columns)
    return x,y,category_names

def tokenize(text):
    """
    Tokenizes the text input from message classification using the Lemmnatizer
    
    input = text
    
    output = clean text 
    
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Builds a grid search model for prediction based off of the 
    a multi output classifier and a K n neighbors classifier
    
    output = prediction model
    
    """
    pipeline = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(KNeighborsClassifier()))])
    parameters = {
        'vectorizer__ngram_range': ((1, 1), (1, 2)),
        'tfidf__norm' : ('l1', 'l2'),
        'clf__estimator__leaf_size': (20, 30, 50),
        'clf__n_jobs': (1, 2, 3),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,  verbose = 3 )
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
     Evaluates a model by fitting the model to the test data 
     and then print the result compared to the prediction
     
     input 
         model = model under evaluation
         x_test = test data for x variable
         y_test = test data for y variable
         category_names = names of the y variables
        
       output : print statements with evaluation results 
    """
    
    
    
    print("Best parameters set found on development set:")
    print()
    print(model.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    print()
    
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_pred = model.predict(X_test)
    for index ,column in enumerate(Y_test, 0):
        f1 = classification_report(Y_test[column].values, [row[index] for row in y_pred])
        print( column)
        print(f1)


def save_model(model, model_filepath):
    """
    Saves the model to a picle file
    
    input 
        model - prediction model
        model_filepath - where the model will be saved
    

    """
    
    
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