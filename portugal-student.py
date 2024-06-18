import requests, zipfile
from io import StringIO
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00356/student.zip'
# データをurlから取得
r = requests.get(url, stream=True)
# zipファイルを読み込み、展開
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

data = pd.read_csv('student-mat.csv')
data.head(10)
# ; を区切り文字に指定
data = pd.read_csv('student-mat.csv', sep=';')
data.head(10)

# データ型
data.info()

# 子はインターネット環境があると、勉強時間が増えるか？
data.groupby('internet')['studytime'].mean()

# ヒストグラム
plt.hist(data['absences'])
plt.grid(True)
# 平均値
print(data['absences'].mean())
# 中央値
print(data['absences'].median())
# 最瀕値
print(data['absences'].mode())
# 分散
print(data['absences'].var())
# 標準偏差
print(data['absences'].std())
# 要約統計量
data['absences'].describe()
# 箱ひげ
plt.boxplot([data['G1'], data['G2'], data['G3']])
plt.grid(True)
# 変動係数　標準偏差 / 平均値
data['absences'].std() / data['absences'].mean()

# 散布図
plt.plot(data['G1'], data['G3'], 'o')
plt.grid(True)

# 共分散行列
np.cov(data['G1'], data['G3'])

# 相関係数　ピアソン
sp.stats.pearsonr(data['G1'], data['G3'])