import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, delimiter=";")

    X = df[['G1', 'G2']]
    y = df['G3']

    y = y.apply(lambda grade: 1 if grade >= 10 else 0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_scaled, X_train, X_test, y_train, y_test


def train_svm(X_train, y_train, kernel='rbf', degree=3, C=1.0, gamma='scale', shrinking=True):
    clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, shrinking=shrinking)
    clf.fit(X_train, y_train)

    return clf


def get_decision_function(clf, X):
    return clf.decision_function(X)


def get_prediction(clf, X):
    return clf.predict(X)


def calculate_accuracy(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def main(file_path='student-math.csv'):
    X_scaled, X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    clf, accuracy = train_svm(X_train, X_test, y_train, y_test)

    return clf, accuracy

