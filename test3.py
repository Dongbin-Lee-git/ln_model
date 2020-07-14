import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
data = pd.read_csv("datasets_insurance.csv") # 데이터 읽기
data.info() # 데이터 정보확인
print(data.describe()) # 데이터 정보확인 2

# ----------------------------------------------------------
# 데이터의 전처리 - smoker : yes(1), no(0)로 변환 , bmi : 30이상(1 : 비만), 30이하(0 : 정상)


def map_smoking(column):
    mapped=[]
    for row in column:
        if row == "yes":
            mapped.append(1)
        else:
            mapped.append(0)
    return mapped


def map_obese(column):
    mapped = []
    for row in column:
        if row > 30:
            mapped.append(1)
        else:
            mapped.append(0)
    return mapped


data["smoker_norm"] = map_smoking(data["smoker"]) # 컬럼추가
data["obese"] = map_obese(data["bmi"]) # 컬럼추가
nonnum_cols=[col for col in data.select_dtypes(include=["object"])]

# -------------------------------------------------------------------
# 데이터 시각화

fig, ax = plt.subplots(len(data.columns)-4, 1, figsize=(3, 25))
ax[0].set_ylabel("charge")
for ind, col in enumerate([i for i in data.columns if i not in ["smoker", "region", "charges", "sex"]]):
    ax[ind].scatter(data[col], data.charges, s=5)
    ax[ind].set_xlabel(col)
    ax[ind].set_ylabel("charge")
plt.show()

# -------------------------------------------------------------------
# 전체 컬럼들의 상관관계 분석위한 도표
corr_vals=[]
collabel=[]
for col in [i for i in data.columns if i not in nonnum_cols]:
    p_val = sp.stats.pearsonr(data[col], data["charges"])
    corr_vals.append(np.abs(p_val[0]))
    print(col, ": ", np.abs(p_val[0]))
    collabel.append(col)
plt.bar(range(1, len(corr_vals) + 1), corr_vals)
plt.xticks(range(1, len(corr_vals) + 1), collabel, rotation=45)
plt.ylabel("Absolute correlation")
plt.show()

# -------------------------------------------------------------------

cols_not_reg3 = ['age', 'obese', 'smoker_norm']

# ----------------------------------------------------------------------
kf = KFold(n_splits=18, random_state=1, shuffle=True)
intercepts = []
mses = []
coefs = []

for train_index, test_index in kf.split(data[cols_not_reg3]):
    lr = linear_model.LinearRegression()
    lr.fit(data[cols_not_reg3].iloc[train_index], data["charges"].iloc[train_index])
    lr_predictions = lr.predict(data[cols_not_reg3].iloc[test_index])
    lr_mse = mean_squared_error(data["charges"].iloc[test_index], lr_predictions)
    intercepts.append(lr.intercept_)
    coefs.append(lr.coef_)
    mses.append(lr_mse)

# ---------------------------------------------------------------------
rmses = [x**.5 for x in mses]
avg_rmse=np.mean(rmses)
avg_intercept=np.mean(intercepts)
age_coefs=[]
obesity_coefs=[]
smoking_coefs=[]
for vals in coefs:
    age_coefs.append(vals[0])
    obesity_coefs.append(vals[1])
    smoking_coefs.append(vals[2])
age_coef=np.mean(age_coefs)
obesity_coef=np.mean(obesity_coefs)
smoking_coef=np.mean(smoking_coefs)
print("a: ",age_coef," b: ",obesity_coef," c: ",smoking_coef," intercept: ",avg_intercept)

# -----------------------------------------------------------------------
def calculate_insurance(age,obesity,smoking):
    y=(age_coef*age)+(obesity_coef*obesity)+(smoking_coef*smoking)+avg_intercept
    return y

print(calculate_insurance(34,1,1))