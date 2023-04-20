import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from logistic_regression import linear_predict, log_reg_train, plot_predictions
import numpy as np
import pylab as plt

data = pd.read_csv('Cleaned_Data.csv')

# Extract sentences and sentiment labels
sentences = data['sentence']
sentiment = data['sentiment']

# Encode the sentiment labels
sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
sentiment = data['sentiment'].map(sentiment_map)

# Vectorize the sentences using bag-of-words model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, sentiment, test_size=0.2, random_state=42)

X_train = X_train.toarray()
X_test = X_test.toarray()

_ = log_reg_train(X_train, y_train,
              {'weights': np.random.randn(X_train.shape[0] * 3)}, check_gradient=True)
#
# # for is_noisy in (False, True):
# #     model[is_noisy] = {'weights': np.zeros((num_dim, num_classes))}
# #
# #     model[is_noisy] = log_reg_train(X_train_bow, y_train_encoded[is_noisy], model[is_noisy])
# #
# #     train_predictions = linear_predict(X_train_bow, model[is_noisy])
# #     train_accuracy[is_noisy] = np.sum(train_predictions == y_train_encoded[is_noisy]) / num_train
# #
# #     test_predictions = linear_predict(X_test_bow, model[is_noisy])
# #     test_accuracy[is_noisy] = np.sum(test_predictions == y_test_encoded[is_noisy]) / num_test
# #
# # print("Train Accuracy for Separable Data: %f" % train_accuracy[False])
# # print("Test Accuracy for Separable Data: %f" % test_accuracy[False])
# # print("Train Accuracy for Noisy Data: %f" % train_accuracy[True])
# # print("Test Accuracy for Noisy Data: %f" % test_accuracy[True])
