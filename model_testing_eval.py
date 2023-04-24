import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from logistic_regression import LogisticRegression

data = pd.read_csv('Training Data/Cleaned_Data.csv')


# Encode the sentiment labels
sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
sentiment = data['sentiment'].map(sentiment_map)

# Vectorize the sentences using bag-of-words model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['sentence'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, sentiment, test_size=0.2, random_state=42)

X_train = X_train.toarray()
X_test = X_test.toarray()

# Instantiate the logistic regression model
logreg = LogisticRegression()

# Train the model
logreg.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg.predict(X_test)

# Evaluate the model

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Classification Report:')
print(classification_report(y_test, y_pred, zero_division=1))
