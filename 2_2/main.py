"""用 Elastic Net 预测 Netflix 用户评分。先运行 download_data.py。"""
from pathlib import Path
import os
import warnings
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore", message="Found unknown categories in columns.*", category=UserWarning)
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / 'ratings_sample.csv')
assert df['rating'].between(1, 5).all()
assert not df.duplicated(['user_id', 'movie_id']).any()
date = pd.to_datetime(df['date'])
X = df[['user_id', 'movie_id']].astype(str)
X['rating_year'] = date.dt.year
X['month_sin'] = np.sin(2 * np.pi * date.dt.month / 12)
X['month_cos'] = np.cos(2 * np.pi * date.dt.month / 12)
y = df['rating'].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ID 是类别，不能把其数值大小当成偏好大小。
# 训练折内出现次数不足5次的用户合为一组，减少单次评分用户导致的过拟合。
prep = ColumnTransformer([
    ('id', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=5, drop='first'),
     ['user_id', 'movie_id']),
    ('time', 'passthrough', ['rating_year', 'month_sin', 'month_cos'])
], sparse_threshold=1.0)
pipe = Pipeline([
    ('encode', prep),
    ('scale', StandardScaler(with_mean=False)),
    ('model', ElasticNet(max_iter=10000, tol=1e-4))
])
cv = KFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, {
    'model__alpha': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    'model__l1_ratio': [0.1, 0.5, 0.9]
}, cv=cv, scoring='neg_mean_squared_error', n_jobs=2, return_train_score=False)
search.fit(X_train, y_train)
best = search.best_estimator_
pred = best.predict(X_test)
baseline = np.full(len(y_test), y_train.mean())

def metrics(a):
    return {'RMSE': float(np.sqrt(mean_squared_error(y_test, a))),
            'MAE': float(mean_absolute_error(y_test, a)),
            'R2': float(r2_score(y_test, a))}

summary = {
    'samples': len(df), 'users': int(df.user_id.nunique()),
    'movies': int(df.movie_id.nunique()), 'train': len(y_train), 'test': len(y_test),
    'rating_mean': float(y.mean()), 'rating_std': float(y.std()),
    'date_min': df.date.min(), 'date_max': df.date.max(),
    'best_params': search.best_params_,
    'cv_RMSE': float(np.sqrt(-search.best_score_)),
    'Elastic_Net': metrics(pred), 'Mean_baseline': metrics(baseline),
    'Elastic_Net_clipped_1_5': metrics(np.clip(pred, 1, 5)),
    'test_unseen_user_fraction': float((~X_test.user_id.isin(X_train.user_id)).mean()),
    'test_unseen_movie_fraction': float((~X_test.movie_id.isin(X_train.movie_id)).mean())
}
names = best.named_steps['encode'].get_feature_names_out()
coef = best.named_steps['model'].coef_
table = pd.DataFrame({'feature': names, 'standardized_coefficient': coef})
table['absolute_coefficient'] = table.standardized_coefficient.abs()
table.sort_values('absolute_coefficient', ascending=False).to_csv(ROOT / 'coefficients.csv', index=False)
summary['features'] = len(coef)
summary['nonzero_features'] = int(np.sum(np.abs(coef) > 1e-8))
(ROOT / 'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
pd.DataFrame(search.cv_results_).to_csv(ROOT / 'cv_results.csv', index=False)
pd.Series(df.rating.value_counts().sort_index()).to_csv(ROOT / 'rating_distribution.csv')
print(json.dumps(summary, indent=2), flush=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df.rating.value_counts().sort_index().plot.bar(ax=axes[0], color='#e50914', rot=0)
axes[0].set(title='Rating distribution (sample)', xlabel='Rating', ylabel='Count')
axes[1].hexbin(y_test, pred, gridsize=22, mincnt=1, cmap='Reds')
axes[1].plot([1, 5], [1, 5], 'k--')
axes[1].set(title='Test: observed vs predicted', xlabel='Observed rating', ylabel='Predicted rating')
axes[2].hist(y_test.to_numpy() - pred, bins=30, color='#444444')
axes[2].set(title='Test residuals', xlabel='Observed minus predicted', ylabel='Count')
fig.tight_layout()
fig.savefig(ROOT / 'analysis.png', dpi=170)
plt.close(fig)

# 在训练集上绘制固定 l1_ratio 的系数路径，不用于重新选择测试集最优模型。
Z = best.named_steps['scale'].transform(best.named_steps['encode'].transform(X_train))
alphas = np.logspace(-4, 0, 30)
paths = np.array([ElasticNet(alpha=a,
    l1_ratio=search.best_params_['model__l1_ratio'], max_iter=10000, tol=1e-4
).fit(Z, y_train).coef_ for a in alphas])
indices = np.argsort(np.abs(coef))[-10:]
fig, ax = plt.subplots(figsize=(9, 5))
for i in indices:
    ax.plot(alphas, paths[:, i], label=names[i].replace('id__', ''))
ax.set_xscale('log')
ax.axvline(search.best_params_['model__alpha'], color='black', linestyle='--')
ax.set(title='Elastic Net: top 10 coefficient paths', xlabel='alpha', ylabel='Standardized coefficient')
ax.legend(fontsize=7, loc='best')
fig.tight_layout()
fig.savefig(ROOT / 'coefficient_path.png', dpi=170)
plt.close(fig)
