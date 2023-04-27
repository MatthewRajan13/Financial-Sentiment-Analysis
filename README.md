# Financial-Sentiment-Analysis #
This project is designed to train 4 different models (Logistic Regression, Multi-layer Perceptron Algorithm, Multi-layer Perceptron Network, Recurrent Neural Network) for sentiment analysis of financial information. The motivation behind this project is to see if financial news sentiment could be a viable trading strategy. After training each model, they will be backtested against the S&P 500 using Positive, Negative, and Neutral predictions on news articles as trading signals. 

## How to Run ##
1. To try this out, please run ```model_testing_eval.py``` to generate and save the models
2. Then you can run ```S&P_Sentiment_Backtest.py``` to see how the saved model wil trade over time! The default model is set to RNN, to switch between models change the model variable to any of the following = [logreg, mlp_algo, mlp_network, rnn]

## Model Accuracy ##
![image](https://user-images.githubusercontent.com/89418442/234877402-80988be9-4908-46ca-83aa-2005dcf58227.png)

## Model Trading Performance ##
### Logistic Regression ##
![image](https://user-images.githubusercontent.com/89418442/234881813-25c1a926-02d8-430b-97ca-6377b47933d2.png)

### Multi-Layer Perceptron Algorithm ###
![image](https://user-images.githubusercontent.com/89418442/234881889-b3b1666b-b8a5-4814-8301-7c1d7e966f63.png)

### Multi-Layer Perceptron Network ###
![image](https://user-images.githubusercontent.com/89418442/234881986-c593b159-be5a-4134-b9e6-a15ff86089c7.png)

### Recurrent Neural Network
![image](https://user-images.githubusercontent.com/89418442/234882101-d3d91262-fcc2-4584-b58f-a0136e82d888.png)
