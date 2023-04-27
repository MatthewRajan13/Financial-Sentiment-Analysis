import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import yfinance as yf
import plotly.graph_objects as go


def main():
    model = 'mlp_pytorch'  # [logreg, mlp_numpy, mlp_pytorch, rnn]
    orig_data = pd.read_csv('Training Data/S&P_News.csv')
    data = preprocess(orig_data)
    predictions = get_Predictions(data, orig_data, model)
    data = fill_dataset(predictions)
    data = backtest(data)
    plot_performance(data, model)


def preprocess(data):
    data = data['Title']

    # Vectorize the sentences using bag-of-words model
    vectorizer = CountVectorizer()
    data = vectorizer.fit_transform(data)

    # Normalize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data.toarray())

    return torch.tensor(data)


def combine_columns(row):
    if row['Negative'] == 1:
        return 'Negative'
    elif row['Neutral'] == 1:
        return 'Neutral'
    elif row['Positive'] == 1:
        return 'Positive'


def get_Predictions(data, orig_data, model='rnn'):
    with open(model + '_model.pickle', 'rb') as f:
        loaded_model = pickle.load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create a tensor of zeros with the expected input size
    padded_input = torch.zeros(100, 7127)
    padded_input[:, :416] = data

    if model == 'logreg' or model == 'mlp_numpy':
        predictions = loaded_model.predict(padded_input.numpy())
        predictions = pd.DataFrame(predictions, columns=['Prediction'])
        predictions['Prediction'] = predictions['Prediction'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
    else:
        predictions = loaded_model.forward(padded_input.to(device)).cpu().detach().numpy()
        max_indices = np.argmax(predictions, axis=1)
        one_hot_encoded = np.zeros_like(predictions)
        one_hot_encoded[np.arange(len(predictions)), max_indices] = 1

        predictions = pd.DataFrame(one_hot_encoded, columns=['Negative', 'Neutral', 'Positive'])

        predictions['Prediction'] = predictions.apply(combine_columns, axis=1)
        predictions = predictions['Prediction']

    combined_df = pd.concat([orig_data, predictions], axis=1)
    combined_df = combined_df.drop([combined_df.columns[0], combined_df.columns[2]], axis=1)

    return combined_df


def fill_dataset(data):
    data = data.drop('Title', axis=1)
    data['Date'] = pd.to_datetime(data['Published']).dt.date
    data = data.drop('Published', axis=1)

    all_dates = pd.date_range(start=data['Date'].min(), end=data['Date'].max(), freq='D')
    filled = pd.DataFrame({'filled': all_dates})

    filled['filled'] = pd.to_datetime(filled['filled']).dt.date
    filled = filled.iloc[::-1].reset_index(drop=True)

    for index, row in filled.iterrows():
        date_value = row['filled']
        matching_row = data[data['Date'] == date_value]
        if not matching_row.empty:
            filled.loc[index, 'sentiment'] = matching_row.iloc[0]['Prediction']

    filled['sentiment'].fillna(value='Neutral', inplace=True)

    filled = filled.set_index('filled')

    spy_data = yf.download('SPY', start='2019-12-01')
    spy_data.index = spy_data.index.date
    spy_data = pd.DataFrame(spy_data['Adj Close'])

    backtest_df = pd.merge(filled, spy_data, left_index=True, right_index=True, how='inner')
    backtest_df = backtest_df.iloc[::-1]

    return backtest_df


def backtest(data):
    # Initialize variables for tracking portfolio and trading signals
    portfolio = 1
    balance = 0
    portfolio_value = []

    for index, row in data.iterrows():
        sentiment = row["sentiment"]
        close_price = row["Adj Close"]

        if sentiment == "Positive" and portfolio == 0:
            portfolio = 1
            balance -= close_price
        elif sentiment == "Negative" and portfolio == 1:
            portfolio = 0
            balance += close_price

        if portfolio == 1:
            portfolio_value.append(balance + close_price)
        else:
            portfolio_value.append(balance)

    data['Portfolio'] = portfolio_value
    data.columns = ['Sentiment', 'S&P', 'Portfolio']

    return data


def plot_performance(data, model):
    fig = go.Figure()
    fig.add_trace(go.Scatter(go.Scatter(x=data.index, y=data['S&P'],
                                        mode='lines', name='S&P 500')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Portfolio'], mode='lines',
                             name='Sentiment Analysis'))

    fig.update_layout(
        xaxis=dict(showline=True, showgrid=True, showticklabels=True, linecolor='white', linewidth=2),
        yaxis=dict(title_text='Price (USD)', titlefont=dict(family='Rockwell', size=12, color='white'),
                   showline=True, showgrid=True, showticklabels=True, linecolor='white', linewidth=2,
                   ticks='outside', tickfont=dict(family='Rockwell', size=12, color='white')),
        showlegend=True, template='plotly_dark')

    annotations = [dict(xref='paper', yref='paper', x=0.0, y=1.05, xanchor='left', yanchor='bottom',
                        text='S&P 500 vs Sentiment Analysis ({})'.format(model.upper()), font=dict(family='Rockwell', size=26, color='white'),
                        showarrow=False)]
    fig.update_layout(annotations=annotations)

    fig.show()


if __name__ == "__main__":
    main()
