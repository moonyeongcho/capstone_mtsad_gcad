import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class smdDataLoader_AD_Interaction:
    """
    smdDataLoader_AD에 interaction term을 추가한 버전.
    voting으로 선택된 변수 쌍의 곱을 새 feature로 추가.

    추가되는 feature 예시 (SMAP, voting=[24, 20, 14]):
        x_24 * x_20, x_24 * x_14, x_20 * x_14
    원래 25개 → 28개
    """

    def __init__(
          self, data, batch_size, seq_len, pred_len, feature_type, target='OT',
          interaction_pairs=None  # [(24, 20), (24, 14), (20, 14)] 형태
        ):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.target = target
        self.target_slice = slice(0, None)
        self.interaction_pairs = interaction_pairs if interaction_pairs is not None else []

        self._read_data()

    def _add_interaction_terms(self, df):
        """
        선택된 쌍의 곱을 새 컬럼으로 추가.

        Args:
            df : pandas DataFrame, shape (T, N)

        Returns:
            df : interaction term이 추가된 DataFrame, shape (T, N + len(pairs))
        """
        df = df.copy()
        for (i, k) in self.interaction_pairs:
            col_name = f"interact_{i}x{k}"
            col_i = df.iloc[:, i].values
            col_k = df.iloc[:, k].values
            df[col_name] = col_i * col_k
        return df

    def _read_data(self):
        """Load raw data, add interaction terms, split datasets."""
        df_raw = pd.read_csv(self.data + '/train.csv', index_col=0)
        df_test_raw = pd.read_csv(self.data + '/test.csv', index_col=0)

        df = df_raw
        df_test_labels = df_test_raw.iloc[:, -1]
        df_test_value = df_test_raw.iloc[:, :-1]

        # interaction term 추가
        df = self._add_interaction_terms(df)
        df_test_value = self._add_interaction_terms(df_test_value)

        # split train/valid/test
        n = len(df)
        train_end = int(n * 0.8)
        val_end = n

        train_df = df[:train_end]
        val_df = df[train_end - self.seq_len : val_end]
        test_df = df_test_value

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.test_labels = df_test_labels
        self.n_feature = self.train_df.shape[-1]

        print(f"원래 feature 수: {df_raw.shape[-1]}")
        print(f"추가된 interaction term: {len(self.interaction_pairs)}개")
        print(f"최종 feature 수: {self.n_feature}")

    def _make_dataset(self, data, shuffle=True, testing=False, test_labels=None):
        data = np.array(data, dtype=np.float32)

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(data[:, self.target_slice], dtype=torch.float32)

        if testing:
            test_labels = np.array(test_labels, dtype=np.float32)
            test_labels = torch.tensor(test_labels, dtype=torch.float32)

        if testing:
            return DataLoader(
                torch.utils.data.Subset(
                    CustomDataset(data_x, data_y, self.seq_len, self.pred_len,
                                  testing=True, test_labels=test_labels),
                    range(len(data_x) - self.seq_len - self.pred_len + 1)
                ),
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True
            )
        else:
            return DataLoader(
                torch.utils.data.Subset(
                    CustomDataset(data_x, data_y, self.seq_len, self.pred_len),
                    range(len(data_x) - self.seq_len - self.pred_len + 1)
                ),
                batch_size=self.batch_size,
                shuffle=shuffle,
                drop_last=True
            )

    def get_train(self, shuffle=True):
        return self._make_dataset(self.train_df, shuffle=shuffle)

    def get_val(self):
        return self._make_dataset(self.val_df, shuffle=False)

    def get_test(self):
        return self._make_dataset(self.test_df, shuffle=False,
                                   testing=True, test_labels=self.test_labels)


class CustomDataset(Dataset):
    def __init__(self, data_x, data_y, seq_len, pred_len,
                 testing=False, test_labels=None):
        self.data_x = data_x
        self.data_y = data_y
        self.test_labels = test_labels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.testing = testing

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, idx):
        if self.testing:
            return (
                self.data_x[idx : idx + self.seq_len],
                self.data_y[idx + self.seq_len : idx + self.seq_len + self.pred_len],
                self.test_labels[idx : idx + self.seq_len + self.pred_len]
            )
        else:
            return (
                self.data_x[idx : idx + self.seq_len],
                self.data_y[idx + self.seq_len : idx + self.seq_len + self.pred_len]
            )
