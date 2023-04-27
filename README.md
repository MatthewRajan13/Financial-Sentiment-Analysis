# Financial-Sentiment-Analysis #
This project is designed to train 3 different algorithms (Logistic Regression, Multi-layer Perceptron, RNN) for sentiment analysis on financial information. The motivation behind this project is to see if news sentiment could be a viable trading strategy. After training each model, they will be backtested against the S&P 500 using Positive, Negative, and Neutral predictions on news articles as trading signals. 

## How to Run ##
To try this out, please run ```model_testing_eval.py``` to generate and save the models. Then you can run ```S&P_Sentiment_Backtest.py``` to see how the saved model wil trade over time!
