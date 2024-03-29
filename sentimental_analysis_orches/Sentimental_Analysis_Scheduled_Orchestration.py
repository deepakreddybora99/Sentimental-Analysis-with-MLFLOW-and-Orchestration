
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)



def split_inputs_output(data, inputs, output):
    """
    Split features and target variables.
    """
    x = data[inputs]
    y = data[output]
    return x, y




def split_train_test(x, y, test_size=0.25, random_state=0):
    """
    Split data into train and test sets.
    """
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def preprocess(x_train, x_test):
    """
    Vectorize the text data.
    """
    vectorizer = CountVectorizer()
    x_train_cleaned = vectorizer.fit_transform(x_train)
    x_test_cleaned = vectorizer.transform(x_test)
    return x_train_cleaned, x_test_cleaned



def train_model(x_train_cleaned, y_train, hyperparameters):
    """
    Training the machine learning model.
    """
    clf = LogisticRegression()
    grid_search = GridSearchCV(clf, hyperparameters, cv=5)
    grid_search.fit(x_train_cleaned, y_train)
    return grid_search



def evaluate_model(model, x_train_cleaned, y_train, x_test_cleaned, y_test):
    """
    Evaluating the model.
    """
    y_train_pred = model.predict(x_train_cleaned)
    y_test_pred = model.predict(x_test_cleaned)

    train_score = metrics.accuracy_score(y_train, y_train_pred)
    test_score = metrics.accuracy_score(y_test, y_test_pred)
    
    return train_score, test_score



def workflow(data_path):
    DATA_PATH = data_path
    INPUTS = "Review text"
    OUTPUT = 'Ratings'
    HYPERPARAMETERS = {
        'C': [10],
        'max_iter': [100]
    }
    
    # Load data
    yonex_data = load_data(DATA_PATH)

    # Identify Inputs and Output
    x, y = split_inputs_output(yonex_data, INPUTS, OUTPUT)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = split_train_test(x, y)

    # Preprocess the data
    x_train_cleaned, x_test_cleaned = preprocess(x_train, x_test)

    # Build a model
    model = train_model(x_train_cleaned, y_train, HYPERPARAMETERS)
    
    # Evaluation
    train_score, test_score = evaluate_model(model, x_train_cleaned, y_train, x_test_cleaned, y_test)
    
    print("Train Score:", train_score)
    print("Test Score:", test_score)



if __name__ == "__main__":
    workflow(data_path="data/yonex_data.csv")


# ### Building a Prefect workflow



from prefect import task, flow

@task
def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

@task
def split_inputs_output(data, inputs, output):
    """
    Split features and target variables.
    """
    x = data[inputs]
    y = data[output]
    return x, y

@task
def split_train_test(x, y, test_size=0.25, random_state=0):
    """
    Split data into train and test sets.
    """
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

@task
def preprocess(x_train, x_test):
    """
    Vectorize the text data.
    """
    vectorizer = CountVectorizer()
    x_train_cleaned = vectorizer.fit_transform(x_train)
    x_test_cleaned = vectorizer.transform(x_test)
    return x_train_cleaned, x_test_cleaned

@task
def train_model(x_train_cleaned, y_train, hyperparameters):
    """
    Training the machine learning model.
    """
    clf = LogisticRegression()
    grid_search = GridSearchCV(clf, hyperparameters, cv=5)
    grid_search.fit(x_train_cleaned, y_train)
    return grid_search

@task
def evaluate_model(model, x_train_cleaned, y_train, x_test_cleaned, y_test):
    """
    Evaluating the model.
    """
    y_train_pred = model.predict(x_train_cleaned)
    y_test_pred = model.predict(x_test_cleaned)

    train_score = metrics.accuracy_score(y_train, y_train_pred)
    test_score = metrics.accuracy_score(y_test, y_test_pred)
    
    return train_score, test_score



@flow(name="Logistic Regression Workflow")
def workflow():
    DATA_PATH = "data/yonex_data.csv"
    INPUTS = "Review text"
    OUTPUT = 'Ratings'
    HYPERPARAMETERS = {
        'C': [10],
        'max_iter': [100]
    }
    
    # Load data
    yonex_data = load_data(DATA_PATH)

    # Identify Inputs and Output
    x, y = split_inputs_output(yonex_data, INPUTS, OUTPUT)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = split_train_test(x, y)

    # Preprocess the data
    x_train_cleaned, x_test_cleaned = preprocess(x_train, x_test)

    # Build a model
    model = train_model(x_train_cleaned, y_train, HYPERPARAMETERS)
    
    # Evaluation
    train_score, test_score = evaluate_model(model, x_train_cleaned, y_train, x_test_cleaned, y_test)
    
    print("Train Score:", train_score)
    print("Test Score:", test_score)


if __name__=='__main__':
    workflow.serve(name="my-first-deployment",
                    cron=" * 8-10 * * 7")




