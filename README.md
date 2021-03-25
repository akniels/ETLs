# ETLs

This repository contains a complete web app that is designed to show how to implement predictive analytics with disaster response message data. 

## The Dataset
Two datasets exist in this project;
* Message dataset
* categories dataset

The message data contains messages that are sent into the disaster response entity needing to be parsed into various categories. The categoires show how the items are parsed afterwards by an agent to the various grouping. Our plan is to make the agents life easier by building a web app that shows the predictions for the messages coming into the system, so that it is a quick spot check and send by the agent. This shows the compelete web app that will help individuals respond quicker in disaters. The complete python web app, datasets, and processes are shown in the data source. Feel free to download the whole folder and run. 

### Instructions:
Here are the instruction for the web app. 

1. Download the complete repository
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3002/

### Files in repository
* app
    - | - template
    - | |- master.html # main page of web app
    - | |- go.html # classification result page of web app
    - |- run.py # Flask file that runs app
* data
    - |- disaster_categories.csv # data to process
    - |- disaster_messages.csv # data to process
    - |- process_data.py
    - |- DisasterResponse.db # database to save clean data to
* models
    - |- train_classifier.py
    - |- classifier.pkl # saved model
* README.md
* ETL Pipeline Preparation
* ML Pipeline Preparation

The Jupyter notebooks in the file show examples of how to build and ETL pipleine for disaster response datasets. The [ETL Pipeline](https://github.com/akniels/ETLs/blob/master/ETL%20Pipeline%20Preparation.ipynb) builds the model and stores the model in a database. The [ML Pipeline](https://github.com/akniels/ETLs/blob/master/ML%20Pipeline%20Preparation.ipynb) takes the data and stores that data into into a model for machine learning. 

# Acknowledgements

* https://stackabuse.com/scikit-learn-save-and-restore-models/
* udacity.com (data science nanodegree)
