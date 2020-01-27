"""
author : Julien SEZNEC
Download raw data from R6A - Yahoo! Front Page Today Module User Click Log Dataset, version 1.0 (1.1 GB)
(https://webscope.sandbox.yahoo.com/catalog.php?datatype=r)
and put the content of the unzip dataset 'R6' in ./data/R6A/
Then, you can run this script which outputs arm's reward plot and data for each day in ./data/Reward. 
"""
import pandas as pd
import logging
from datetime import datetime
from matplotlib import pyplot as plt
import gzip
import os

logging.getLogger('matplotlib.font_manager').disabled = True
plt.style.use('seaborn-colorblind')
plt.style.use('style.mplstyle')

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)


def merge_datasets():
    """
    Merge the 10 files and keep the three useful columns ['timestamp', 'article_id', 'click'] in out.csv
    """
    out = open("./data/R6A/out.csv", "a+")
    out.write('timestamp article_id click\n')
    for j in range(10):
        f = gzip.open("./data/R6A/ydata-fp-td-clicks-v1_0.20090" + str(501 + j) + '.gz', "r")
        for i, line in enumerate(f.readlines()):
            if i % 10000 == 0:
                logging.info("./data/R6A/ydata-fp-td-clicks-v1_0.20090" + str(501 + j) + ' : ' + str(i))
            out.write(line.decode('utf-8').split('|')[0][:-1] + '\n')


def prepare_dataset():
    """
    Download out.csv, cast the right datatypes and convert date
    :return: Dataframe - columns =  ['timestamp', 'article_id', 'click', 'date']
    """
    df = pd.read_csv('./data/R6A/out.csv', usecols=['timestamp', 'article_id', 'click'], sep=' ')
    df = df.astype({'timestamp': int, 'article_id': int, 'click': bool})
    df['date'] = df.timestamp.apply(lambda dt: datetime.fromtimestamp(dt))
    return df


def compute_average(df):
    """
    Compute 1) rolling average of N sample ; 2) Average for each timestamp of the rolling average
    (one value per timestamp)
    :return: dataframe of the average click probabilities.  columns = article_id ; index = date
    """
    rolling_mean = df.groupby('article_id').click.rolling(30000, min_periods=14000, center=True).mean()
    rolling_mean.index = rolling_mean.index.droplevel(0)
    rolling_mean.name = 'click_mean'
    df_with_rolling = df.join(pd.DataFrame(rolling_mean), how='inner')
    final_mean = df_with_rolling.groupby(['article_id', 'date']).click_mean.mean().unstack().transpose()
    return final_mean


def compute_traffic(df):
    """
    :return: Series. The total traffic (user count) for each timestamp.
    """
    return df.date.value_counts().sort_index()


def split_dataset(df, traffic, freq=10):
    """
    Split the dataset in 10 days with one value per round (i.e. 10 user).
    :return: list of 10 Series (cast article_id on a list of mean rewards)
    """
    dfs = []
    for i in range(1, 11):
        if len(str(i)) == 1:
            dfs.append(
                df[(df.index >= '2009-05-0' + str(i) + ' 00:00:00') & (df.index < '2009-05-0' + str(i) + ' 12:00:00')])
        else:
            dfs.append(
                df[(df.index >= '2009-05-' + str(i) + ' 00:00:00') & (df.index < '2009-05-' + str(i) + ' 12:00:00')])
    dfs = [df[df.columns[~df.isnull().any()]] if len(df) else pd.DataFrame() for df in dfs]
    dfs = [d.apply(lambda row: row.apply(lambda x: [x])) for d in dfs]
    dfs = [
        d.apply(lambda col: col * traffic.loc[col.index])
            .sum()
            .apply(lambda liste: pd.Series(liste[::freq])) for d in dfs
    ]
    for i, d in enumerate(dfs):
        d.to_csv('./data/Reward/reward_data_day_%s.csv' % (i + 1))
    return dfs


def plot_reward(df, i):
    """
    Plot the reward functions.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    df.apply(lambda data: ax.plot(range(len(data)), data), axis=1)
    ax.set_ylim([0, 0.075])
    ax.set_xlabel('Round (t)')
    ax.set_ylabel("Arms Average Reward")
    ax.set_title("Day %s - $K = %s$" % (i + 1, len(df)), y= 1.03)
    fig.savefig("./data/Reward/reward_plot_day%s.pdf" % (i + 1))


if __name__ == '__main__':
    os.makedirs('./data/Reward/', exist_ok=True)
    if not os.path.isfile('./data/R6A/out.csv'):
        merge_datasets()
    logging.info("Prepare dataset")
    df = prepare_dataset()
    logging.info("Compute average")
    df_average = compute_average(df)
    logging.info("Compute traffic")
    traffic = compute_traffic(df)
    logging.info("Split dataset")
    dfs = split_dataset(df_average, traffic)
    for i, df in enumerate(dfs):
        plot_reward(df, i)
