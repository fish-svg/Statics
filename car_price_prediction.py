# 個人の属性から収入が５万ドルを超えるか否か分類する
# ロジスティック回帰
import requests
import pandas as pd
import io

# データを取得
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
res = requests.get(url).content

# 取得したデータをDataFrameオブジェクトとして読み込み
adult = pd.read_csv(io.StringIO(res.decode('utf-8')), header=None)

# 列ラベルの追加
adult.columns =['age','workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
               'native-country','flg-50K']

# データの型の確認と欠損値
print(adult.shape)
print(adult.isnull().sum().sum())

# ０と１に変換
# 新たな列を生成：もし＞５０Ｋならば１をそれ以外は0を代入
adult['fin_flg'] = adult['flg-50K'].map(lambda x: 1 if x == ' >50K' else 0)
# 確認
adult.groupby('fin_flg').size()

# モデル構築と評価
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 説明変数
X = adult[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss']]
y = adult['fin_flg']

# 訓練データとテストデータ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# モデルの初期化と学習
model = LogisticRegression()
model.fit(X_train, y_train)

# 評価
print(model.score(X_test, y_test))

# 各係数
print(model.coef_)

# 標準化による予測精度の向上
# 0 or 1 にする
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# モデルの初期化と学習
model = LogisticRegression()
model.fit(X_train_std, y_train)

# 評価
print(model.score(X_test_std, y_test))

# 各係数
print(model.coef_)
