import re

import pandas as pd


def transform_money(x: str):
    if not isinstance(x, str):
        return x
    x = x.strip("€")
    if x.endswith('M'):
        x = float(x.strip('M')) * 1000
    elif x.endswith('K'):
        x = float(x.strip('K'))
    else:
        x = float(x)
    return x


def transform_height(x: str):
    if isinstance(x, str) and "'" in x:
        f, s = x.split("'", maxsplit=1)
        return int(f) * 12 + int(s)
    return int(x)


def transform_rate(x: str):
    if x == "High":
        return 3
    elif x == "Medium":
        return 2
    elif x == "Low":
        return 1
    else:
        print(x)


def str_preprocessing(x):
    if isinstance(x, str):
        matcher = re.match("\w+-(\d+)", x)
        if matcher:
            return int(matcher.group(1))
        matcher = re.match("(\w+)([+-])(\w+)", x)
        if matcher:
            if matcher.group(2) == '+':
                return int(matcher.group(1)) + int(matcher.group(3))
            elif matcher.group(2) == '-':
                return int(matcher.group(1)) - int(matcher.group(3))
            else:
                print(x)
        return int(x)
    return x


def preprocess_fifa18(filename="fifa18.csv"):
    df = pd.read_csv(filename, header=0, index_col=0)
    df = df.drop(
        labels=['ID', 'Name', 'Photo', 'Flag', 'Club Logo', 'Club', 'Nationality', 'CAM', 'CB', 'CDM', 'CF', 'CM',
                'LAM', 'LB', 'LCB', 'LCM', 'LDM', 'LF', 'LM', 'LS', 'LW', 'LWB', 'RAM', 'RB', 'RCB', 'RCM', 'RDM', 'RF',
                'RM', 'RS', 'RW', 'RWB', 'ST', 'Preferred Positions', 'Special'],
        axis=1
    )
    df["Value"] = df["Value"].apply(transform_money)
    df["Wage"] = df["Wage"].apply(transform_money)
    df = df.applymap(str_preprocessing)
    df.to_csv(f"pre_{filename}", index=False)


def preprocess_fifa19(filename="fifa19.csv"):
    df = pd.read_csv(filename, header=0, index_col=0)
    print(df.shape)
    df = df.drop(
        labels=['Joined', 'Loaned From', 'Contract Valid Until', 'Real Face', "Photo", "Club", "Club Logo", "Flag",
                'ID', 'Name', 'Nationality', 'Release Clause', 'Jersey Number',
                "LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM",
                "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB", "CB", "RCB", "RB", 'Special'],
        axis=1
    )
    df = df.dropna(axis=0)
    df["Value"] = df["Value"].apply(transform_money)
    df["Wage"] = df["Wage"].apply(transform_money)
    df["Weight"] = df["Weight"].apply(
        lambda x: x.strip("lbs") if isinstance(x, str) and x.endswith("lbs") else x).astype("int32")
    df["Height"] = df["Height"].apply(transform_height)
    work_rates = df["Work Rate"].str.split("/", n=1, expand=True)
    df["Work Rate1"] = work_rates[0].str.strip().apply(transform_rate)
    df["Work Rate2"] = work_rates[1].str.strip().apply(transform_rate)
    df = df.drop("Work Rate", axis=1)
    df = pd.get_dummies(df)
    print(df.shape)
    df.to_csv(f"pre_{filename}", encoding="utf-8", index=False)


if __name__ == '__main__':
    preprocess_fifa18()
    preprocess_fifa19()
