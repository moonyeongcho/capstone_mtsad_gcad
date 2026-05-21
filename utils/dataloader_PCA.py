# 데이터를 읽고 전처리해서 모델이 바로 먹을 수 있는 형태로 바꿉니다. 
# 정확히는 train.csv, test.csv를 읽고, 표준화하고, 시계열을 seq_len 길이의 입력과 pred_len 길이의 정답으로 잘라서 DataLoader로 만들어 줍니다. 
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def _scale_df(df, scaler):
    data = scaler.transform(df.values)
    return pd.DataFrame(data, index=df.index, columns=df.columns)


def _apply_pca_if_enabled(train_df, val_df, test_df, pca_enabled, pca_n_components=None, pca_var_ratio=None):
    if not pca_enabled:
        return train_df, val_df, test_df, None

    # [PCA-1] PCA는 표준화된 학습 분할에 대해서만 학습하고,
    # 검증/테스트 데이터에는 같은 투영을 재사용합니다.
    n_components = pca_var_ratio if pca_var_ratio is not None else pca_n_components
    pca = PCA(n_components=n_components)
    pca.fit(train_df.values)

    columns = [f"pc{i + 1}" for i in range(pca.n_components_)]

    def transform_df(df):
        data = pca.transform(df.values)
        return pd.DataFrame(data, index=df.index, columns=columns)

    return transform_df(train_df), transform_df(val_df), transform_df(test_df), pca


class SwatDataLoader_AD:
    """Generate data loader from raw data."""

    def __init__(
          self, data, batch_size, seq_len, pred_len, feature_type, target='OT', stride=1,
          pca_enabled=False, pca_n_components=None, pca_var_ratio=None
        ):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.target = target
        self.target_slice = slice(0, None)
        self.stride = stride
        self.pca_enabled = pca_enabled
        self.pca_n_components = pca_n_components
        self.pca_var_ratio = pca_var_ratio
        self.pca = None

        # [PCA-2] PCA로 투영한 뒤에는 원래 타깃 컬럼과의 일대일 대응이 사라지므로
        # PCA는 완전한 다변량 설정(feature_type='M')에서만 허용합니다.
        if self.pca_enabled and self.feature_type != 'M':
            raise ValueError("PCA can only be used with feature_type='M' because target feature mapping is lost after projection.")

        self._read_data()

    def _read_data(self):
        """Load raw data and split datasets."""
        df_raw = pd.read_csv(self.data+'/train.csv', index_col=0)
        df_test_raw = pd.read_csv(self.data+'/test.csv', index_col=0)

        # 학습 데이터의 마지막 열은 라벨이므로 제외하고, 테스트에서는 값과 라벨을 분리한다.
        df = df_raw.iloc[:,:-1]
        df_test_labels = df_test_raw.iloc[:,-1]
        df_test_value = df_test_raw.iloc[:,:-1]

        # 학습 데이터를 train/val로 나누고 테스트는 별도 파일을 사용한다.
        n = len(df)

        train_end = int(n * 0.8)
        val_end = n
        # val_end = n - int(n * 0.2)

        train_df = df[:train_end]
        val_df = df[train_end - self.seq_len : val_end]
        test_df = df_test_value

        # 스케일링 기준은 반드시 학습 구간으로만 맞춘다.
        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        train_df = _scale_df(train_df, self.scaler)
        val_df = _scale_df(val_df, self.scaler)
        test_df = _scale_df(test_df, self.scaler)

        # [PCA-3] PCA 전에 표준화를 적용해
        # 스케일이 큰 센서만 주성분을 지배하지 않도록 합니다.
        self.train_df, self.val_df, self.test_df, self.pca = _apply_pca_if_enabled(
            train_df,
            val_df,
            test_df,
            self.pca_enabled,
            self.pca_n_components,
            self.pca_var_ratio,
        )

        # self.train_df = train_df
        # self.val_df = val_df
        # self.test_df = test_df   

        self.test_labels = df_test_labels

        self.n_feature = self.train_df.shape[-1]

    def _make_dataset(self, data, shuffle=True, testing=False, test_labels=None, stride=1):
        # DataFrame을 float32 텐서로 바꾸고 슬라이딩 윈도우 데이터셋을 생성한다.
        data = np.array(data, dtype=np.float32)

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(data[:, self.target_slice], dtype=torch.float32)

        if testing:
            test_labels = np.array(test_labels, dtype=np.float32)
            test_labels = torch.tensor(test_labels, dtype=torch.float32)

        if testing:
            return DataLoader(
                torch.utils.data.Subset(
                    CustomDataset(data_x, data_y, self.seq_len, self.pred_len, testing=True, test_labels=test_labels),
                    # 테스트 시 stride를 두어 평가 간격을 조절할 수 있다.
                    range(0,len(data_x) - self.seq_len - self.pred_len + 1, stride)
                ),
                batch_size=self.batch_size, 
                shuffle=shuffle,
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

    # def inverse_transform(self, data):
    #     return self.scaler.inverse_transform(data)

    def get_train(self, shuffle=True):
        return self._make_dataset(self.train_df, shuffle=shuffle)

    def get_val(self):
        return self._make_dataset(self.val_df, shuffle=False)

    def get_test(self):
        return self._make_dataset(self.test_df, shuffle=False, testing=True, test_labels=self.test_labels,stride=self.stride)

class smdDataLoader_AD:
    """Generate data loader from raw data."""

    def __init__(
          self, data, batch_size, seq_len, pred_len, feature_type, target='OT',
          pca_enabled=False, pca_n_components=None, pca_var_ratio=None
        ):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.target = target
        self.target_slice = slice(0, None)
        self.pca_enabled = pca_enabled
        self.pca_n_components = pca_n_components
        self.pca_var_ratio = pca_var_ratio
        self.pca = None

        # [PCA-2] PCA로 투영한 뒤에는 원래 타깃 컬럼과의 일대일 대응이 사라지므로
        # PCA는 완전한 다변량 설정(feature_type='M')에서만 허용합니다.
        if self.pca_enabled and self.feature_type != 'M':
            raise ValueError("PCA can only be used with feature_type='M' because target feature mapping is lost after projection.")

        self._read_data()

    def _read_data(self):
        """Load raw data and split datasets."""
        df_raw = pd.read_csv(self.data+'/train.csv', index_col=0)
        df_test_raw = pd.read_csv(self.data+'/test.csv', index_col=0)

        # SMD는 train 파일에 라벨 열이 없다고 가정하고 전체를 값으로 사용한다.
        df = df_raw
        df_test_labels = df_test_raw.iloc[:,-1]
        df_test_value = df_test_raw.iloc[:,:-1]

        # 학습 데이터를 train/val로 분할한다.
        n = len(df)

        train_end = int(n * 0.8)
        val_end = n
        # val_end = n - int(n * 0.2)

        train_df = df[:train_end]
        val_df = df[train_end - self.seq_len : val_end]
        test_df = df_test_value

        # 학습 구간 통계량으로만 표준화한다.
        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        train_df = _scale_df(train_df, self.scaler)
        val_df = _scale_df(val_df, self.scaler)
        test_df = _scale_df(test_df, self.scaler)

        # [PCA-3] PCA 전에 표준화를 적용해
        # 스케일이 큰 센서만 주성분을 지배하지 않도록 합니다.
        self.train_df, self.val_df, self.test_df, self.pca = _apply_pca_if_enabled(
            train_df,
            val_df,
            test_df,
            self.pca_enabled,
            self.pca_n_components,
            self.pca_var_ratio,
        )

        # self.train_df = train_df
        # self.val_df = val_df
        # self.test_df = test_df

        self.test_labels = df_test_labels

        self.n_feature = self.train_df.shape[-1]

    def _make_dataset(self, data, shuffle=True, testing=False, test_labels=None, train_average=False):
        # 모델 입력과 타깃 텐서를 만든 뒤 슬라이딩 윈도우 형태로 묶는다.
        data = np.array(data, dtype=np.float32)

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(data[:, self.target_slice], dtype=torch.float32)

        if testing:
            test_labels = np.array(test_labels, dtype=np.float32)
            test_labels = torch.tensor(test_labels, dtype=torch.float32)

        if testing:
            return DataLoader(
                torch.utils.data.Subset(
                    CustomDataset(data_x, data_y, self.seq_len, self.pred_len, testing=True, test_labels=test_labels),
                    range(len(data_x) - self.seq_len - self.pred_len + 1)
                ),
                batch_size=self.batch_size, 
                # batch_size=1,
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
        return self._make_dataset(self.test_df, shuffle=False, testing=True, test_labels=self.test_labels)

class CustomDataLoader:
    """Generate data loader from raw data."""

    def __init__(
          self, data, batch_size, seq_len, pred_len, feature_type, target='OT',
          pca_enabled=False, pca_n_components=None, pca_var_ratio=None
        ):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.target = target
        self.target_slice = slice(0, None)
        self.pca_enabled = pca_enabled
        self.pca_n_components = pca_n_components
        self.pca_var_ratio = pca_var_ratio
        self.pca = None

        # [PCA-2] PCA로 투영한 뒤에는 원래 타깃 컬럼과의 일대일 대응이 사라지므로
        # PCA는 완전한 다변량 설정(feature_type='M')에서만 허용합니다.
        if self.pca_enabled and self.feature_type != 'M':
            raise ValueError("PCA can only be used with feature_type='M' because target feature mapping is lost after projection.")

        self._read_data()

    def _read_data(self):
        """Load raw data and split datasets."""
        df_raw = pd.read_csv(self.data)

        # feature_type에 따라 단변량/다변량 예측 대상을 결정한다.
        df = df_raw.set_index('date')
        if self.feature_type == 'S':
            df = df[[self.target]]
        elif self.feature_type == 'MS':
            target_idx = df.columns.get_loc(self.target)
            self.target_slice = slice(target_idx, target_idx + 1)

        # 데이터셋 종류에 따라 분할 규칙을 다르게 적용한다.
        n = len(df)
        if self.data.stem.startswith('ETTm'):
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = val_end + 4 * 30 * 24 * 4
        elif self.data.stem.startswith('ETTh'):
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = val_end + 4 * 30 * 24
        else:
            train_end = int(n * 0.7)
            val_end = n - int(n * 0.2)
            test_end = n
        train_df = df[:train_end]
        val_df = df[train_end - self.seq_len : val_end]
        test_df = df[val_end - self.seq_len : test_end]

        # 학습 구간 기준 표준화를 적용한다.
        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        train_df = _scale_df(train_df, self.scaler)
        val_df = _scale_df(val_df, self.scaler)
        test_df = _scale_df(test_df, self.scaler)

        # [PCA-3] PCA 전에 표준화를 적용해
        # 스케일이 큰 센서만 주성분을 지배하지 않도록 합니다.
        self.train_df, self.val_df, self.test_df, self.pca = _apply_pca_if_enabled(
            train_df,
            val_df,
            test_df,
            self.pca_enabled,
            self.pca_n_components,
            self.pca_var_ratio,
        )
        self.n_feature = self.train_df.shape[-1]

    def _make_dataset(self, data, shuffle=True):
        # 일반 시계열 예측용 슬라이딩 윈도우 데이터셋을 생성한다.
        data = np.array(data, dtype=np.float32)

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(data[:, self.target_slice], dtype=torch.float32)
            
        return DataLoader(
            torch.utils.data.Subset(
                CustomDataset(data_x, data_y, self.seq_len, self.pred_len),
                range(len(data_x) - self.seq_len - self.pred_len + 1)
            ),
            batch_size=self.batch_size, 
            shuffle=shuffle
        )

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_train(self, shuffle=True):
        return self._make_dataset(self.train_df, shuffle=shuffle)

    def get_val(self):
        return self._make_dataset(self.val_df, shuffle=False)

    def get_test(self):
        return self._make_dataset(self.test_df, shuffle=False)
    

class CustomDataset(Dataset):
    def __init__(self, data_x, data_y, seq_len, pred_len, testing=False, test_labels=None):
        self.data_x = data_x
        self.data_y = data_y
        self.test_labels  = test_labels
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.testing = testing

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, idx):
        '''data_x.shape: (batch_size, input_len, num_features)
           data_y.shape: (batch_size, pred_len, num_features)
           test_labels.shape(batch_size, input_len+pred_len)
        '''
        if self.testing:
            # 테스트 시에는 입력/타깃과 함께 해당 구간의 라벨 시퀀스도 반환한다.
            return self.data_x[idx : idx + self.seq_len], self.data_y[idx + self.seq_len : idx + self.seq_len + self.pred_len], self.test_labels[idx : idx + self.seq_len + self.pred_len]
        else:
            # 학습/검증 시에는 입력과 예측 대상만 반환한다.
            return self.data_x[idx : idx + self.seq_len], self.data_y[idx + self.seq_len : idx + self.seq_len + self.pred_len]
