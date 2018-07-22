import datetime as dt
import sys
from datetime import datetime
from datetime import timedelta
import requests
from time import sleep
import pandas as pd

class GDAX(object):
    def __init__(self, pair):
        self.pair = pair
        self.uri = 'https://api.gdax.com/products/{pair}/candles'.format(pair=self.pair)

    def fetch(self, start, end, granularity):
        data = []
        # We will fetch the candle data in windows of maximum 100 items.
        delta = timedelta(minutes=granularity * 100)

        slice_start = start
        while slice_start != end:
            slice_end = min(slice_start + delta, end)
            data += self.request_slice(slice_start, slice_end, granularity)
            slice_start = slice_end

        # I prefer working with some sort of a structured data, instead of
        # plain arrays.
        data_frame = pd.DataFrame(data=data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        data_frame.set_index('time', inplace=True)
        return data_frame

    def request_slice(self, start, end, granularity):
        # Allow 3 retries (we might get rate limited).
        retries = 3
        for retry_count in range(0, retries):
            # From https://docs.gdax.com/#get-historic-rates the response is in the format:
            # [[time, low, high, open, close, volume], ...]
            response = requests.get(self.uri, {
              'start': start.isoformat(),
              'end': end.isoformat(),
              'granularity': granularity * 60  # GDAX API granularity is in seconds.
            })

            if response.status_code != 200 or not len(response.json()):
                print(response.json())
                if retry_count + 1 == retries:
                    raise Exception('Failed to get exchange data for ({}, {})!'.format(start, end))
                else:
                    # Exponential back-off.
                    sleep(1.5 ** retry_count)
            else:
                # Sort the historic rates (in ascending order) based on the timestamp.
                result = sorted(response.json(), key=lambda x: x[0])
                return result

    @staticmethod
    def __date_to_iso8601(date):
        return '{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}'.format(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=date.hour,
            minute=date.minute,
            second=date.second)

def get_latest_date_from_btc_daily():

    # Read in current csv
    df = pd.read_csv('btc-daily.csv')
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Get last date of data retrieved
    max_date = df.time.max().to_pydatetime()

    # Compute from/to date (two days before and after last date retrieved)
    from_date = max_date - timedelta(days=2)
    to_date = max_date + timedelta(days=2)

    return from_date, to_date

if __name__ == '__main__':

    print('Getting latest data...')

    from_date, to_date = get_latest_date_from_btc_daily()
    print('From date: ')
    print(from_date)
    print('To date: ')
    print(to_date)

    data_frame = GDAX('BTC-USD').fetch(from_date, to_date, 1440)

    file_name = 'btc-daily.csv'

    print('Writing to: ')
    print(file_name)

    with open(file_name, 'a') as f:
        data_frame.to_csv(f, header=False)

    print('Done.')
