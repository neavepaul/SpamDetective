import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

def train_and_save_model():
    # Load the dataset
    df = pd.read_csv("Data/spam.csv", encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

    # Map labels to binary values
    df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
    df['message'] = df['v2']
    df.drop(['v1', 'v2'], axis=1, inplace=True)
    
    # Features and labels
    X = df['message']
    y = df['label']

    # Extract features with CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Evaluate the model
    accuracy = clf.score(X_test, y_test)
    print(f"Model accuracy: {accuracy}")

    # Save the trained model to a file
    model_filename = 'spam_classifier_model.pkl'
    joblib.dump(clf, model_filename)
    joblib.dump(cv, 'count_vectorizer.pkl')
    print(f"Trained model saved as {model_filename}")

# Call the function to train and save the model
train_and_save_model()
