import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module): 
    """Residual block of TSMixer."""
    
    def __init__(self, input_shape, dropout, ff_dim):
        super(ResBlock, self).__init__()
        # 파라미터 정의

        # Temporal Linear
        self.norm1 = nn.BatchNorm1d(input_shape[0]*input_shape[1]) # BatchNorm1d는 2D 입력만 받으므로 flatten(= seq_len * N)
        self.linear1 = nn.Linear(input_shape[0], input_shape[0]) # 시간 축끼리 섞는
        self.dropout1 = nn.Dropout(dropout)
        
        # Feature Linear (n_feature 차원을 ff_dim으로 넓히고)
        self.norm2 = nn.BatchNorm1d(input_shape[0]*input_shape[1])
        self.linear2 = nn.Linear(input_shape[-1], ff_dim) # input_shape[-1]은 N, N-->ff_dim: 변수축 확장
        self.dropout2 = nn.Dropout(dropout)
        
        # Feature Mixing 2단계: (ff_dim=1024 -> n_feature=51으로 다시 )
        self.linear3 = nn.Linear(ff_dim, input_shape[-1]) # ff_dim-->N: 변수축 복원
        self.dropout3 = nn.Dropout(dropout)

    # 여기가 내가 손으로 적어서 정리해놓은 부분
    def forward(self, x):
        inputs = x # residual connection을 위해 원본 입력 저장
        
        # Temporal Linear
        x = self.norm1(torch.flatten(x, 1, -1)).reshape(x.shape) # 2D를 위한 flatten
        x = torch.transpose(x, 1, 2) # (batch, seq_len, n_feature) -> (batch, n_feature, seq_len)
        x = F.relu(self.linear1(x))
        x = torch.transpose(x, 1, 2) # 다시 원래대로 transpose
        x = self.dropout1(x) 
        
        res = x + inputs

        # Feature Linear
        x = self.norm2(torch.flatten(res, 1, -1)).reshape(res.shape) 
        # 여긴 feature mixer라 transpose 안함
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        
        x = self.linear3(x)
        x = self.dropout3(x)

        return x + res


# https://github.com/ts-kim/RevIN/blob/master/RevIN.py
class RevIN(nn.Module): ## 변수별(독립)로 정규화/역정규화 수행 (시간축을 기준으로)
    def __init__(self, num_features: int, eps=1e-5, affine=True): 
        # eps: 분모가 0이 되는 걸 방지하는 작은 값
        # affine: True--> 학습 가능한 파라미터(weight, bias)를 사용, False--> 사용하지 않음 (정규화만 평균0, 표준편차1)
        
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str, target_slice=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_slice)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features)) # affine_weight 1로 초기화
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))  # affine_bias 0으로 초기화

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x): # ReVIN 정규화 방식 (평균, 표준편차로 계산하고 affine이면 affine_weight, affine_bias 곱함)
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        if self.affine:
            x = x - self.affine_bias[target_slice]
            x = x / (self.affine_weight + self.eps*self.eps)[target_slice]
        x = x * self.stdev[:, :, target_slice]
        x = x + self.mean[:, :, target_slice]
        return x