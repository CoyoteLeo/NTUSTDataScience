#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import requests #引入函式庫
from bs4 import BeautifulSoup
import re

data = pd.read_csv("CompleteDataset.csv", header=0, skip_blank_lines=True)

data["ID"] = data["ID"].apply(str)

col = ["season", "team", "competition", "minutes played", "appearances", "lineups", "substitute in", "substitue out", "substitutes on bench", "goal", "yellow card", "yellow 2nd/RC", "red card", "playerID"]
count = 0 #紀錄丟到df的第幾行了
df = pd.DataFrame(columns=col)

f = open('error.txt', 'a+')  #紀錄error

for all_id in range(0, 17981): #0~17980個球員
    url = "https://sofifa.com/player/" + data["ID"][all_id]+"/live"

    resp = requests.get(url) #回傳為一個request.Response的物件
    soup = BeautifulSoup(resp.text, 'html.parser')
    if resp.status_code != 200:
        f.write(str(all_id)+" : connect error\n")
        continue
    if soup == None:
        f.write(str(all_id)+" : soup error\n")
        continue
    if soup.find('table', "real-career no-link table table-hover") == None:
        f.write(str(all_id)+" : empty page or no career chart error\n")
        continue
    if soup.find('table', "real-career no-link table table-hover").tbody == None:
        f.write(str(all_id)+" : no career chart error\n")
        continue
    rows = soup.find('table', "real-career no-link table table-hover").tbody.find_all('tr')  #找到該球員所有資料

    list1 = []
    mainlist=[]
    
    for tdrow in rows:  #先看每年
        for tdrowdata in tdrow:  #看年的每個column
            tdrowdata = tdrowdata.text.strip()  #從該column的html拿出字
            list1.append(tdrowdata) #蒐集滿一row資料
        list1.append(data["ID"][all_id])
        if len(list1) != 14:
            continue
        mainlist.append(list1)  #將所有row併成一個表
        list1 = []  #清空temp row
    
    if len(list1) != 0:
        f.write(str(all_id)+" : has career chart but no data error\n")
        continue
    for i in range(len(mainlist)):  #將表傳進df
        df.loc[i + count] = mainlist[i]
    count += len(mainlist)
    print(all_id)

df.to_csv('Result.csv')
f.close()
print("Done")
