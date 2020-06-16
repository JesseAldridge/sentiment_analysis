import os, glob, time, pickle, csv
from datetime import datetime

import twint
from nltk import tokenize

import generate_classifier

def build_get_sentiment():
  with open("naivebayes.pickle", "rb") as f:
    classifier = pickle.load(f)

  def get_sentiment(text):
    tokens = generate_classifier.remove_noise(tokenize.word_tokenize(text))
    result = classifier.prob_classify(dict((token, True) for token in tokens))
    return result.prob('Positive')

  return get_sentiment
get_sentiment = build_get_sentiment()

class Tweet:
  def __init__(self, text):
    self.body = text
    self.sentiment = get_sentiment(self.body)

def download_tweets(symbol, tweets_dir):
  config = twint.Config()
  config.Search = '${}'.format(symbol)
  config.Limit = 100
  config.Store_csv = True
  config.Output = out_path = os.path.join(tweets_dir, "{}.csv".format(symbol))
  config.Hide_output = True
  twint.run.Search(config)
  return out_path

LIMIT = None

def main():
  while True:
    sentiment_dir = os.path.expanduser("~/stock_sentiment")
    tweets_dir = os.path.join(sentiment_dir, "tweets")
    if not os.path.exists(tweets_dir):
      os.makedirs(tweets_dir)

    symbol_sentiments = []
    for path in glob.glob(os.path.expanduser('~/stock_data/*.csv'))[:LIMIT]:
      symbol = path.rsplit('.csv', 1)[0].rsplit('/', 1)[1]

      print('grabbing tweets for:', symbol)
      tweets_path = download_tweets(symbol, tweets_dir)

      with open(tweets_path) as f:
        tweet_rows = list(csv.DictReader(f))
      tweets = [Tweet(tweet_row.get('tweet')) for tweet_row in tweet_rows]

      sum_ = 0
      for tweet in tweets:
        sum_ += tweet.sentiment
      sentiment = sum_ / len(tweets)
      symbol_sentiments.append((symbol, sentiment))

    symbol_sentiments.sort(key=lambda t: t[1])
    timestamp = '_'.join(str(datetime.utcnow()).split()).rsplit('.', 1)[0].replace(':', '-')
    path = os.path.join(sentiment_dir, '{}'.format(timestamp)) + '.csv'
    with open(path, 'w') as f:
      writer = csv.writer(f, lineterminator='\n')
      writer.writerow(('symbol', 'sentiment'))
      for row in symbol_sentiments:
        writer.writerow(row)

    print("{}, done, sleeping for 24 hours".format(datetime.utcnow()))
    time.sleep(60 * 60 * 24)

if __name__ == '__main__':
  main()