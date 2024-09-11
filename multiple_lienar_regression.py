# 重回帰
# 自動車の属性から価格を予測する
import requests, zipfile
import io

# 自動車データの取得
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
res = requests.get(url).content

# 取得したデータをDataFrame オブジェクトとして読み込み
auto = pd.read_csv(io.StringIO(res.decode('utf-8')),header=None)

# データの列にラベルを設定
# データの列にラベルを設定
auto.columns =['symboling','normalized-losses','make','fuel-type' ,'aspiration','num-of-doors',
                            'body-style','drive-wheels','engine-location','wheel-base','length','width','height',
                            'curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore',
                            'stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']

# ？データのカウント
auto = auto[['price', 'horsepower', 'width', 'height']]
auto.isin(['?']).sum()

# ？をNaNに置換して削除
auto = auto.replace('?', np.nan).dropna()

# 型の確認
auto.dtypes

# オブジェクト型を数値型へ変換
auto = auto.assign(price=pd.to_numeric(auto.price))
auto = auto.assign(horsepower=pd.to_numeric(auto.horsepower))
auto.dtypes

# 各変数間の相関
auto.corr()

# データの分割のモジュール
from sklearn.model_selection import train_test_split

# 重回帰モデル構築のためのインポート
from sklearn.linear_model import LinearRegression

# 目的変数：price，説明変数：それ以外
X = auto.drop('price', axis=1)
y = auto['price']

# 訓練データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# 重回帰クラスの初期化と学習
model = LinearRegression()
model.fit(X_train, y_train)

# 決定係数を表示
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

# 回帰係数と切片を表示
print(pd.Series(model.coef_, index = X.columns))
print(model.intercept_)
