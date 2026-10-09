"""下载 Netflix Prize 的课程实验子集。原数据对应 Kaggle netflix-inc/netflix-prize-data。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = 'https://raw.githubusercontent.com/deesethu/Netflix-Dataset/master/'
IDS = sorted(np.random.default_rng(42).choice(np.arange(1, 17771), 80, replace=False).tolist())

def fetch(movie):
    url = BASE + f'training_set/training_set/mv_{movie:07d}.txt'
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                lines = response.read().decode('utf-8').strip().splitlines()
            if lines[0] != f'{movie}:':
                raise ValueError('Unexpected movie header')
            rows = [line.split(',') for line in lines[1:]]
            # 每部电影最多随机抽2000条，避免热门电影占据整个样本。
            if len(rows) > 2000:
                indices = np.random.default_rng(42 + movie).choice(len(rows), 2000, replace=False)
                rows = [rows[i] for i in sorted(indices)]
            return [[int(u), movie, int(r), d] for u, r, d in rows]
        except Exception:
            if attempt == 2:
                raise

if __name__ == '__main__':
    all_rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, rows in enumerate(pool.map(fetch, IDS), 1):
            all_rows.extend(rows)
            if i % 10 == 0:
                print(f'Downloaded {i}/80 movies', flush=True)
    data = pd.DataFrame(all_rows, columns=['user_id', 'movie_id', 'rating', 'date'])
    data.to_csv(ROOT / 'ratings_sample.csv', index=False)
    (ROOT / 'sample_movie_ids.txt').write_text(','.join(map(str, IDS)), encoding='utf-8')
    print(f'Saved {len(data)} real ratings')
