import os

import pandas as pd


def sanity_check(data: pd.DataFrame) -> (int, pd.DataFrame):
    invalid_rows = 0
    for idx, row in data.iterrows():
        if not isinstance(row['text'], str):
            data.drop(idx, inplace=True)
            invalid_rows += 1
    return invalid_rows, data


def main():
    file = os.path.join('./Data/train.csv')
    data = pd.read_csv(file)
    invalid_rows, data = sanity_check(data)
    print(f"Cleared {invalid_rows} non-string reviews")
    pd.DataFrame.to_csv(data, file)


if __name__ == '__main__':
    main()
