import pandas as pd
import requests

with open('words.txt', 'r') as file:
    english_words = set(word.strip().lower() for word in file.readlines())


def process_text_file(input_file):
    data = []
    with open(input_file, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()
        for line in lines:
            sentence, sentiment = line.strip().split("@")
            data.append((sentence, sentiment))
    return data


def filter_english_words(sentence):
    filtered_words = []
    words = sentence.split()
    for word in words:
        if word.strip().lower() in english_words:
            filtered_words.append(word)
    return ' '.join(filtered_words)


def clean(data):
    data.sentence = data.sentence.str.lower()

    # Remove special characters and digits
    data.sentence = data.sentence.str.replace('[^a-zA-Z\s]', '', regex=True)
    data.sentence = data.sentence.str.replace('\d+', '', regex=True)

    data.sentence = data['sentence'].apply(filter_english_words)

    data.sentence = data.sentence.str.replace(r'\s+', ' ', regex=True)
    data.sentence = data.sentence.str.strip()

    return data


def main():

    files = [('Sentences_AllAgree.txt', 'Allagree.csv'), ('Sentences_75Agree.txt', '75agree.csv'),
             ('Sentences_50Agree.txt', '50agree.csv'), ('Sentences_66Agree.txt', '66agree.csv'),]

    texts = []
    for input_file in files:
        texts.append(input_file[0])

    datas = []
    for text in texts:
        data = pd.DataFrame(process_text_file(text))
        data.columns = ['sentence', 'sentiment']
        datas.append(data)

    cleaned = []
    for i, data in enumerate(datas):
        clean_data = clean(data)
        cleaned.append(clean_data)
        clean_data.to_csv(files[i][1])

    # Get Kaggle dataset
    kaggle = pd.read_csv('kaggle.csv')
    kaggle.columns = ['sentence', 'sentiment']

    kaggle = clean(kaggle)
    kaggle = kaggle.replace('', pd.NA)
    kaggle = kaggle.dropna()
    kaggle.to_csv('CleanKaggle.csv')

    agree_all = cleaned[0]
    agree_75 = cleaned[1]

    appended_all = pd.concat([kaggle, agree_all], axis=0)
    appended_all.reset_index(drop=True, inplace=True)
    appended_all.to_csv('AppendedAll.csv')

    appended_75 = pd.concat([kaggle, agree_75], axis=0)
    appended_75.reset_index(drop=True, inplace=True)
    appended_75.to_csv('Appended75.csv')


if __name__ == '__main__':
    main()
