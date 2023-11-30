import os

import pandas as pd
from sklearn.model_selection import train_test_split


def split(path: str = './data/'):
    training_data = pd.read_csv(os.path.join(path, 'train.csv'))
    train_set, val_set = train_test_split(training_data, train_size=0.8, random_state=42)
    pd.DataFrame.to_csv(train_set, os.path.join('./data/train_val.csv'))
    pd.DataFrame.to_csv(val_set, os.path.join('./data/val_val.csv'))


def main():
    split()


if __name__ == '__main__':
    main()
