import os, glob, time, pickle, csv, asyncio, logging, sys
from datetime import datetime

import twint, aiohttp
from nltk import tokenize

import generate_classifier

def setup_logger():
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  sh = logging.StreamHandler(sys.stdout)
  formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S'
  )
  sh.setFormatter(formatter)
  logger.addHandler(sh)
logger = setup_logger()


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
    logging.info(f'downloading to dir: {sentiment_dir}')

    timestamp = '_'.join(str(datetime.utcnow()).split()).rsplit('.', 1)[0].replace(':', '-')
    sentiment_path = os.path.join(sentiment_dir, '{}'.format(timestamp)) + '.csv'
    symbol_sentiment_rows = []

    csv_paths = glob.glob(os.path.expanduser('~/stock_data/YahooPrices/*.csv'))[:LIMIT]
    for icsv, prices_path in enumerate(csv_paths):
      symbol = prices_path.rsplit('.csv', 1)[0].rsplit('/', 1)[1]

      logging.info(f'{icsv}/{len(csv_paths)}, grabbing tweets for: {symbol}')
      try:
        tweets_path = download_tweets(symbol, tweets_dir)
      except asyncio.TimeoutError:
        logging.info('timeout error')
        continue
      except aiohttp.client_exceptions.ClientConnectorError:
        logging.info('ClientConnectorError')
        continue
      except aiohttp.client_exceptions.ClientPayloadError:
        logging.info('ClientPayloadError')
        continue

      try:
        with open(tweets_path) as f:
          tweet_rows = list(csv.DictReader(f))
      except FileNotFoundError:
        logging.info(f'file not found: {tweets_path}')
        continue
      tweets = [Tweet(tweet_row.get('tweet')) for tweet_row in tweet_rows]

      sentiment = sum(tweet.sentiment for tweet in tweets) / len(tweets)
      symbol_sentiment_rows.append((symbol, sentiment))
      symbol_sentiment_rows.sort(key=lambda t: t[-1])

      with open(sentiment_path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(('symbol', 'sentiment'))
        for row in symbol_sentiment_rows:
          writer.writerow(row)

      logging.info(f'wrote to: {sentiment_path}')
    logging.info("done, sleeping for 24 hours")
    time.sleep(60 * 60 * 24)

if __name__ == '__main__':
  main()
