import requests
import pandas as pd
import bs4 as bs
import csv


def main():
    feed = 'http://www.spglobal.com/spdji/en/rss/rss-details/?rssFeedName=corporate-news'

    news = get_articles(feed)

    news = pd.DataFrame(news)

    news.to_csv("S&P_News.csv")


def get_articles(url):
    r = requests.get(url)
    soup = bs.BeautifulSoup(r.content, features='xml')

    article_list = []
    articles = soup.findAll('item')
    for a in articles:
        title = a.find('title').text
        link = a.find('link').text
        published = a.find('pubDate').text
        article = {
            'Title': title,
            'Link': link,
            'Published': published
        }
        article_list.append(article)

    return article_list


if __name__ == "__main__":
    main()
