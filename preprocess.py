import numpy as np
import pandas as pd


# 計算分數中的+號
def score(x):
    temp = []
    for i in x:
        if isinstance(i, str) and '+' in i:
            temp.append(sum(map(int, i.split('+'))))
        else:
            temp.append(i)
    return np.array(temp)


# 把'month-'去掉
def stripMonth(x):
    temp = []
    for i in x:
        if isinstance(i, str):
            temp.append(int(''.join(c for c in i if c.isdigit())))
        else:
            temp.append(i)
    return np.array(temp)

def process_lable_data():
    # 讀取薪資、市場價值的data, 並去掉錢幣符號
    csv = pd.read_csv('CompleteDataset2019.csv')
    wage = csv.loc[:, ['ID', 'Wage']].set_index('ID')
    wage['Wage'] = wage['Wage'].map(lambda x: float(x.strip('€K ')) * 1000 if 'K' in x else (float(x.strip('€M ')) * 100000) if 'M' in x else (x.strip('€ ')))
    # print(wage)
    value = csv.loc[:, ['ID', 'Value']].set_index('ID')
    value = value['Value'].map(lambda x: float(x.strip('€K ')) * 1000 if 'K' in x else (float(x.strip('€M ')) * 100000) if 'M' in x else (x.strip('€ ')))
    label = pd.concat([wage, value], axis=1)

    return label
    

# preprocessing
def preprocess_player_data():


    # 讀取屬性資料
    csv = pd.read_csv('CompleteDataset.csv', encoding="ISO-8859-1")
    temp = csv.iloc[:, 0:47]
    temp = temp.drop(['Name', 'Photo', 'Flag', 'Club Logo', 'Value', 'Wage', 'Unnamed: 0'], axis=1)
    temp = temp.apply(lambda x: score(x))

    # 補上ID欄位
    ID = csv.iloc[:, 52]
    temp = pd.concat([temp, ID], axis=1)

    # 補上prefer pos欄位並做one hot encoding
    prefer = csv.iloc[:, 63]
    prefer = prefer.T.squeeze().str.split(' ', expand=True).stack()
    prefer = pd.get_dummies(prefer).groupby(level=0).sum().drop([''], axis=1)
    df18 = pd.concat([temp, prefer], axis=1)
    df18_OneHot = pd.get_dummies(data=df18, columns=['Nationality', 'Club', ])
    df18_int = df18_OneHot.apply(lambda x: stripMonth(x)).set_index('ID')

    return df18_int
    

def preprocess():
    train = preprocess_player_data()
    label = process_lable_data()
    res = pd.merge(label, train, on=['ID'])
    res.to_csv('train.csv')
    return res

if __name__ == '__main__':
    preprocess()
