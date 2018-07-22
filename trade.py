import pandas as pd
import numpy as np
from stockstats import StockDataFrame
import datetime as dt
from datetime import datetime
from datetime import timedelta

np.random.seed()

print('Running trade algo...')

df = pd.read_csv('btc-daily.csv')
df['time'] = pd.to_datetime(df['time'], unit='s')
df_original = df

df['date'] = df['time']
del df['time']
df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

df = df.sort_values('Date')
df = df.set_index('Date')
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

df = df[~df.index.duplicated(keep='last')]

# To test against actual values
df_unsmoothed = df

# Smooth close prices
df = df.ewm(com=9.5).mean()

# Calculate target values
def calculate_targets_for_df(the_df):
    # Remove date index
    the_df.reset_index(level=0, inplace=True)

    def calculate_target(row):
        current_row_idx = row.name
        future_row_idx = current_row_idx + 1
        if future_row_idx in the_df.index:
            dif = the_df.iloc[future_row_idx]['Close'] - the_df.iloc[current_row_idx]['Close']
            if dif > 0:
                return 1
            elif dif < 0:
                return -1
            else:
                return 0
        else:
            return 1000 # throw away

    the_df['target'] = the_df.apply(calculate_target, axis=1)
    # Set date as index
    the_df = the_df.set_index('Date')

    return the_df

df = calculate_targets_for_df(df)

df_unsmoothed = calculate_targets_for_df(df_unsmoothed)

# Save target values
target = df['target']
target_unsmoothed = df_unsmoothed['target']

# Delete target column from dataframes
del df['target']
del df_unsmoothed['target']

ss_df = StockDataFrame.retype(df)

ss_df['rsi'] = ss_df['rsi_14']

ss_df['wr'] = ss_df['wr_14']

ss_df['macd'] = ss_df['macd']

# Limit to features
features_df = ss_df[['rsi', 'wr', 'macd']]

# Remove first 26 rows and last row
features_df = features_df.iloc[27:len(features_df)]

# Limit to OHLCV
df = df[['open', 'high', 'low', 'close', 'volume']]

# Remove first 26 rows and last row
df = df.iloc[27:len(df)]

target = target[27:len(target)]

target_unsmoothed = target_unsmoothed[27:len(target_unsmoothed)]

# Join all
df_final = pd.merge(df, features_df, left_index=True, right_index=True)
df_final['target'] = target

# Run if everything checks out
df_use = df_final
df = df_final[:len(df_final)-1]

df_unsmoothed = df_unsmoothed.iloc[27:len(df_unsmoothed)]

df_unsmoothed = df_unsmoothed[:len(df_unsmoothed)-1]

target_unsmoothed = target_unsmoothed[:len(target_unsmoothed)-1]

df_unsmoothed['target'] = target_unsmoothed

from sklearn.ensemble import RandomForestClassifier

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

train, test = df[df['is_train']==True], df[df['is_train']==False]

df_unsmoothed['is_train'] = df['is_train']

train_unsmoothed, test_unsmoothed = df_unsmoothed[df_unsmoothed['is_train']==True], df_unsmoothed[df_unsmoothed['is_train']==False]

features = df.columns[:8]

y = pd.factorize(train['target'])[0]

clf = RandomForestClassifier(n_jobs=2, random_state=0)

clf.fit(train[features], y)

test['target'] = test_unsmoothed['target']

# Map to the actual target sign for each predicted class
target_names = train['target'].unique()
preds = target_names[clf.predict(test[features])]

crosstab_results = pd.crosstab(test['target'], preds, rownames=['Actual Sign'], colnames=['Predicted Sign'])

tn = crosstab_results[-1][-1]
tp = crosstab_results[1][1]
fn = crosstab_results[-1][1]
fp = crosstab_results[1][-1]


not_predicted_at_all = 0
if 0 in crosstab_results[-1].index:
    not_predicted_at_all += crosstab_results[-1][0]
    not_predicted_at_all += crosstab_results[1][0]

total = tn+tp+fn+fp+not_predicted_at_all

print('Accuracy: ')
print((tn+tp)/float(total))
print('\n\n')

# Get yesterday
yesterday = df_use[len(df_use)-1:]
# Get relevant features
yesterday = yesterday[features]

print('Yesterday: ')
print(yesterday)
print('\n\n')

pred = target_names[clf.predict(yesterday)][0]
probs = clf.predict_proba(yesterday)[0]

print('Prediction: ')
print(probs)
print('\n\n')
