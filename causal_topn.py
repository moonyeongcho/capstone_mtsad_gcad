import os
os.makedirs('_multivariate', exist_ok=True)

from collections import Counter
from itertools import combinations
import pandas as pd
import numpy as np
from pathlib import Path

datasets = [
    #('checkpoints/smd_machine-1-1', 'smd'),
    #('checkpoints/smap_T-3',        'smap'),
    #('checkpoints/msl_P-15',        'msl'),
    ('checkpoints/psm',             'psm'),
]

results = {}

for checkpoint_base, dataset_name in datasets:
    print(f'\n{"="*50}')
    print(f'데이터셋: {dataset_name}')
    print(f'{"="*50}')

    checkpoint_dirs = [checkpoint_base] + [f'{checkpoint_base}_{i}' for i in range(2, 11)]

    records = {}
    all_votes = []
    summary_records = []

    for k, cp_dir in enumerate(checkpoint_dirs):
        csv_path = Path(cp_dir) / 'causal_parms.csv'
        if not csv_path.exists():
            print(f'없음: {csv_path}')
            continue

        mat = pd.read_csv(csv_path, index_col=None, header=None).values
        mat_no_diag = mat.copy()
        np.fill_diagonal(mat_no_diag, -np.inf)
        argmax_per_col = mat_no_diag.argmax(axis=0)
        max_per_col = mat_no_diag.max(axis=0)
        all_votes.extend(argmax_per_col.tolist())

        for j in range(mat.shape[1]):
            if j not in records:
                records[j] = {'j열': j}
            records[j][f'i열_{k}'] = argmax_per_col[j]
            records[j][f'최댓값_{k}'] = round(max_per_col[j], 6)

        counter = Counter(argmax_per_col.tolist())
        for i, cnt in counter.most_common():
            summary_records.append({'실험': k, 'i열': i, '득표수': cnt})

    if not records:
        print(f'{dataset_name} checkpoint 없음, 스킵')
        continue

    # per_checkpoint
    cols = ['j열']
    for k in range(10):
        cols += [f'i열_{k}', f'최댓값_{k}']
    cols_available = [c for c in cols if c in list(records[0].keys())]
    df_per = pd.DataFrame(list(records.values()))[cols_available]
    df_per.to_csv(f'_multivariate/{dataset_name}_voting_per_checkpoint.csv', index=False)
    print(f'=> {dataset_name}_voting_per_checkpoint.csv 저장 완료')

    # summary
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(f'_multivariate/{dataset_name}_voting_summary.csv', index=False)
    print(f'=> {dataset_name}_voting_summary.csv 저장 완료')

    # total
    counter_total = Counter(all_votes)
    df_total = pd.DataFrame(counter_total.most_common(), columns=['i열', '득표수'])
    df_total.to_csv(f'_multivariate/{dataset_name}_voting_total.csv', index=False)
    print(f'=> {dataset_name}_voting_total.csv 저장 완료')

    # n별 조합
    results[dataset_name] = {}
    for n in [3, 5, 10]:
        top_n = df_total['i열'].iloc[:n].tolist()
        bottom_n = df_total['i열'].iloc[-n:].tolist()

        top_pairs = list(combinations(top_n, 2))
        bottom_pairs = list(combinations(bottom_n, 2))

        results[dataset_name][n] = {
            'top_n': top_n, 'top_pairs': top_pairs,
            'bottom_n': bottom_n, 'bottom_pairs': bottom_pairs
        }

        print(f'\nn={n} | top{n} i: {top_n} | top nC2 ({len(top_pairs)}쌍): {top_pairs}')
        print(f'n={n} | bottom{n} i: {bottom_n} | bottom nC2 ({len(bottom_pairs)}쌍): {bottom_pairs}')