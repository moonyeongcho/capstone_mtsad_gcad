import numpy as np
import pandas as pd
import os
from pathlib import Path
from itertools import combinations

# -----------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------
MULTIVARIATE_DIR = '_multivariate'
N_LIST = [3, 5]

datasets = [
    ('smd', 'datasets/smd/machine-1-1'),
    ('smap', 'datasets/smap/T-3'),
    ('msl', 'datasets/msl/P-15'),
    ('psm', 'datasets/psm'),  # causal_topn.py 돌린 후 사용
]

# 윈도우 실행(bat) 베이스 명령어
base_cmds = {
    'smd':  'python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.005 --test_stride 1 --n_block 3 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu',
    'smap': 'python main.py --seq_len 70 --pred_len 1 --pd_beta 1 --sample_p 0.2 --sparse_th 0.008 --test_stride 1 --n_block 6 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu',
    'msl':  'python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.002 --test_stride 1 --n_block 5 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu',
    'psm':  'python main.py --seq_len 30 --pred_len 5 --pd_beta 0.5 --sample_p 0.1 --sparse_th 0.005 --test_stride 10 --n_block 2 --ff_dim 128 --dropout 0 --learning_rate 0.0001 --device cpu',
}

# -----------------------------------------------------------------------
# 데이터셋별 처리
# -----------------------------------------------------------------------
for dataset_name, data_dir in datasets:
    print(f'\n{"="*50}')
    print(f'데이터셋: {dataset_name}')
    print(f'{"="*50}')

    total_csv = Path(MULTIVARIATE_DIR) / f'{dataset_name}_voting_total.csv'
    if not total_csv.exists():
        print(f'없음: {total_csv} → 스킵')
        continue

    df_total = pd.read_csv(total_csv)
    all_i = df_total['i열'].tolist()

    # 원본 데이터 로드
    df_train = pd.read_csv(f'{data_dir}/train.csv', index_col=0)
    df_test_raw = pd.read_csv(f'{data_dir}/test.csv', index_col=0)
    df_test = df_test_raw.iloc[:, :-1]
    df_test_labels = df_test_raw.iloc[:, -1]

    # causal matrix 평균 (v2 select용)
    checkpoint_base = f'checkpoints/{dataset_name}_' + data_dir.split('/')[-1]
    checkpoint_dirs = [checkpoint_base] + [f'{checkpoint_base}_{i}' for i in range(2, 11)]

    matrices = []
    for cp_dir in checkpoint_dirs:
        csv_path = Path(cp_dir) / 'causal_parms.csv'
        if not csv_path.exists():
            continue
        mat = pd.read_csv(csv_path, index_col=None, header=None).values
        matrices.append(mat)

    avg_mat = np.mean(np.stack(matrices, axis=0), axis=0)
    N = avg_mat.shape[0]
    mat_no_diag = avg_mat.copy()
    np.fill_diagonal(mat_no_diag, 0)

    candidates = []
    for i in range(N):
        for k in range(i + 1, N):
            strength = mat_no_diag[i][k] + mat_no_diag[k][i]
            if strength > 0:
                candidates.append((i, k, strength))
    candidates.sort(key=lambda x: x[2], reverse=True)

    def add_interaction(df, pairs):
        df = df.copy()
        for (i, k) in pairs:
            df[f'interact_{i}x{k}'] = df.iloc[:, i].values * df.iloc[:, k].values
        return df

    def save_dataset(folder, train, test, labels):
        os.makedirs(folder, exist_ok=True)
        train.to_csv(f'{folder}/train.csv')
        test_with_label = test.copy()
        test_with_label['label'] = labels.values
        test_with_label.to_csv(f'{folder}/test.csv')
        print(f'=> 저장 완료: {folder}')

    # v1: 보팅
    voting_pairs = {}
    for n in N_LIST:
        top_n = all_i[:n]
        bottom_n = all_i[-n:]
        voting_pairs[f'top_n{n}'] = list(combinations(top_n, 2))
        voting_pairs[f'bottom_n{n}'] = list(combinations(bottom_n, 2))

    print('\n[방식1 보팅]')
    for key, pairs in voting_pairs.items():
        print(f'{key}: {pairs}')
        folder = f'{data_dir}_v1/{key}'
        save_dataset(folder, add_interaction(df_train, pairs),
                     add_interaction(df_test, pairs), df_test_labels)

    # v2: select
    select_pairs = {}
    for n in N_LIST:
        select_pairs[f'top_m{n}'] = [(i, k) for i, k, s in candidates[:n]]
        select_pairs[f'bottom_m{n}'] = [(i, k) for i, k, s in candidates[-n:]]

    print('\n[방식2 select]')
    for key, pairs in select_pairs.items():
        print(f'{key}: {pairs}')
        folder = f'{data_dir}_v2/{key}'
        save_dataset(folder, add_interaction(df_train, pairs),
                     add_interaction(df_test, pairs), df_test_labels)

    # run.bat 생성 (윈도우용)
    base_cmd = base_cmds[dataset_name]
    data_stem = data_dir.split('/')[-1]

    # 윈도우 배치 파일(.bat) 헤더 및 주석 문법 적용
    lines1 = ['@echo off\n\n']
    for exp_name in voting_pairs.keys():
        data_path = f'./{data_dir}_v1/{exp_name}'
        name = f'{dataset_name}_{data_stem}_v1_{exp_name}'
        lines1.append(f'REM {exp_name}\n')
        lines1.append(f'{base_cmd} --data "{data_path}" --name "{name}"\n\n')

    with open(f'run_interaction_{dataset_name}_v1.bat', 'w') as f:
        f.writelines(lines1)
    print(f'=> run_interaction_{dataset_name}_v1.bat 생성 완료')

    lines2 = ['@echo off\n\n']
    for exp_name in select_pairs.keys():
        data_path = f'./{data_dir}_v2/{exp_name}'
        name = f'{dataset_name}_{data_stem}_v2_{exp_name}'
        lines2.append(f'REM {exp_name}\n')
        lines2.append(f'{base_cmd} --data "{data_path}" --name "{name}"\n\n')

    with open(f'run_interaction_{dataset_name}_v2.bat', 'w') as f:
        f.writelines(lines2)
    print(f'=> run_interaction_{dataset_name}_v2.bat 생성 완료')