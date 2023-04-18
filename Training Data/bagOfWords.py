import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Read the CSV file
data = pd.read_csv('Cleaned_Data.csv')

# Extract sentences and sentiment labels
sentences = data['Sentences']
sentiment = data['Sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sentences, sentiment, test_size=0.2, random_state=42)

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the training data
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_bow = vectorizer.transform(X_test)

# Initialize the LabelEncoder
encoder = LabelEncoder()

# Fit and transform the training labels
y_train_encoded = encoder.fit_transform(y_train)

# Transform the testing labels
y_test_encoded = encoder.transform(y_test)
