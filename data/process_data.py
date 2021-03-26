import sys


import sys
import os
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the data from two CSVs into a dataframe
    
    inputs : 
        
        Message_filepath - message csv
        
        categories_filepath - Category csv
        
    outputs : Dataframe with merged data
    
    
    
    """
    df = pd.read_csv(messages_filepath)
    df2 = pd.read_csv(categories_filepath)
    new_df = df.merge(df2, left_on='id', right_on='id')
    new_df.drop_duplicates(subset=['id'], inplace=True)
    return new_df

def clean_data(df):
    """
    Cleans the data from dataframe to be shaped to start machine learning prediction
    
    Input = Dataframe
    
    Output - Clean dataframe
    
    """
    
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x : x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype('int32')
    new_categories = pd.concat([categories.reset_index(drop=True),categories.reset_index(drop=True)], axis=1)
    #new_categories = new_categories.drop(['categories'], axis=1)
    categories['related'] = categories['related'].replace([2],1)
    df = df.merge(categories, left_on='id', right_index=True)
    df = df.drop_duplicates()
    return df

def save_data(df, database_filepath):
    """
    Saves the data to a sqllite database
    
    input 
        dataframe : data frame you would like to load
        
        database_filepath : Filepath of where the sqllight database will be saved
    
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
#     if os.path.isfile(database_filename):
#         os.remove(database_filename)
    df.to_sql('Message', engine, index=False, if_exists='replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()