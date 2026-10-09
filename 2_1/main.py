"""9.3 第4题。安装：python -m pip install numpy pandas matplotlib scikit-learn"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / 'Hitters.csv').dropna()
X = pd.get_dummies(df.drop(columns='Salary'), drop_first=True).astype(float)
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
A = scaler.fit_transform(X_train)
B = scaler.transform(X_test)
cv = KFold(n_splits=10, shuffle=True, random_state=42)
alphas = np.logspace(-3, 5, 100)
models = {
    'Ridge': RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_squared_error'),
    'Lasso': LassoCV(alphas=alphas, cv=cv, max_iter=100000, tol=1e-6),
    'Elastic Net': ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.5, 0.9],
                              cv=cv, max_iter=100000, tol=1e-6)
}

def new_model(name, alpha, ratio=0.5):
    if name == 'Ridge':
        return Ridge(alpha=alpha)
    if name == 'Lasso':
        return Lasso(alpha=alpha, max_iter=100000, tol=1e-6)
    return ElasticNet(alpha=alpha, l1_ratio=ratio, max_iter=100000, tol=1e-6)

rows = []
coefficients = {}
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, model) in zip(axes, models.items()):
    model.fit(A, y_train)
    ratio = float(model.l1_ratio_) if name == 'Elastic Net' else 0.5
    alpha = float(model.alpha_)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(B)))
    count = int(np.sum(np.abs(model.coef_) > 1e-8))
    rows.append([name, 'CV-min (course)', alpha,
                 ratio if name == 'Elastic Net' else np.nan, rmse, count])
    coefficients[name] = model.coef_

    # 1-SE 的10折误差：每折只使用该折训练数据拟合标准化器。
    # Elastic Net 固定上面选出的 l1_ratio，仅沿 alpha 方向比较。
    errors = np.zeros((len(alphas), 10))
    for k, (train, valid) in enumerate(cv.split(X_train)):
        fold_scaler = StandardScaler()
        C = fold_scaler.fit_transform(X_train.iloc[train])
        D = fold_scaler.transform(X_train.iloc[valid])
        for i, a in enumerate(alphas):
            m = new_model(name, a, ratio).fit(C, y_train.iloc[train])
            errors[i, k] = mean_squared_error(y_train.iloc[valid], m.predict(D))
    means = errors.mean(axis=1)
    se = errors.std(axis=1, ddof=1) / np.sqrt(10)
    best = int(np.argmin(means))
    threshold = means[best] + se[best]
    one_se = alphas[np.where(means <= threshold)[0][-1]]
    # 同时报告严格按折标准化后的 min，保证1-SE有一致的比较基准。
    for label, a in [('CV-min (fold-scaled)', alphas[best]), ('1-SE', one_se)]:
        m = new_model(name, a, ratio).fit(A, y_train)
        rows.append([name, label, a,
                     ratio if name == 'Elastic Net' else np.nan,
                     np.sqrt(mean_squared_error(y_test, m.predict(B))),
                     int(np.sum(np.abs(m.coef_) > 1e-8))])

    path = np.array([new_model(name, a, ratio).fit(A, y_train).coef_
                     for a in alphas])
    ax.plot(alphas, path)
    ax.axvline(alpha, color='black', linestyle='--', label='course CV-min')
    ax.axvline(one_se, color='gray', linestyle=':', label='1-SE')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_title(name + (f' (l1_ratio={ratio:g})' if name == 'Elastic Net' else ''))
    ax.set_xlabel('alpha (regularization strength)')
    ax.set_ylabel('Coefficient on standardized X')
    ax.legend(fontsize=8)
    pd.DataFrame({'alpha': alphas, 'CV_MSE': means, 'SE': se}).to_csv(
        ROOT / (name.replace(' ', '_') + '_cv.csv'), index=False)

fig.tight_layout()
fig.savefig(ROOT / 'coefficient_paths.png', dpi=180)
plt.close(fig)
result = pd.DataFrame(rows, columns=['Model', 'Rule', 'alpha', 'l1_ratio', 'Test_RMSE', 'Nonzero'])
result.to_csv(ROOT / 'results.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(coefficients, index=X.columns).to_csv(ROOT / 'coefficients.csv', encoding='utf-8-sig')
print(f'Samples: {len(df)}, train: {len(X_train)}, test: {len(X_test)}, features: {X.shape[1]}')
print(result.to_string(index=False))
print('Saved results.csv and coefficient_paths.png')
