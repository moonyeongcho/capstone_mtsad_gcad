# GCAD 기반 다변량 시계열 이상 탐지
Multivariate Time Series Anomaly Detection based on GCAD (AAAI-2025)
---

## 프로젝트 소개

본 프로젝트는 AAAI-2025에 발표된 **GCAD**(*Granger Causality-based Anomaly Detection*)를 재현하고, 성능 향상을 위한 다양한 실험을 수행한 캡스톤 디자인 연구이다.

GCAD는 다변량 시계열 데이터에서 변수 간 Granger causality를 gradient를 통해 동적으로 추정하고, 정상 패턴과의 인과 행렬 편차를 이상 점수로 활용한다.

- 저자 github: [https://github.com/Tc99m/GCAD](https://github.com/Tc99m/GCAD)
- 논문: [AAAI 2025 - Liu et al.](https://ojs.aaai.org/index.php/AAAI/article/view/34096)


### 연구 흐름

1. GCAD 원본 재현 및 baseline 확보
2. 파생변수 추가 실험 (규칙 A/B/C 기반)
3. PCA 적용 실험 및 방향성 검토


### 이상 점수 공식

```
S = Sc + β · St
```

- `Sc`: 인과 패턴 편차 점수 (causal deviation score)
- `St`: 시간 패턴 편차 점수 (time pattern deviation score)
- `β`: `--pd_beta` 인수로 조정

---

### 주요 파일 설명

**`main.py`**  
학습 루프 담당. `patience=2` early stopping, 10회 반복 실험 후 결과(인과 행렬)를 CSV로 저장한다. 실험별 seed는 고정 목록(`experiment_seeds`)에서 순서대로 할당된다.

**`test.py`**  
훈련 완료 후 두 가지 역할을 수행합니다.
- `save_train_mean_causal`: 학습 데이터에서 정상 인과 패턴 행렬(`Anorm`) 추출
- `test`: 테스트 데이터에서 이상 점수 계산 및 ROC / PRC / F1 평가

**`tsmixer.py`**  
TSMixer 기반 예측 모델로 GCAD의 gradient generator 역할. RevIN을 통해 입력을 정규화하고, 복수의 ResBlock을 통과한 뒤 FC layer로 예측값을 출력한다.

**`dataloader.py`**  
SMD / SMAP / MSL는 이미 정규화된 데이터이므로 StandardScaler를 적용하지 않도록 수정했으며, 
세 데이터 모두 smdDataLodaerAD를 사용한다.

---

## 실험 결과

- 모든 실험은 10회 반복 후 평균값을 사용한다. 
- 10번의 반복 실험은 재현성을 위해 시드를 고정하였다.
- 아래 수치는 팀 자체 재현 baseline으로, 논문 원본 수치와 차이가 있다. (시드 미공개 및 코드 부분 공개)

### 1. 파생변수 추가 실험 (규칙 A/B/C)

- MLP backpropagation gradient로 인과 행렬을 구성한 뒤, 규칙별로 변수 쌍을 선택하여 element-wise 곱을 새 feature로 추가하는 방식이다. 
- Random 선택을 대조군으로 설정하여 규칙의 유효성을 검증했다.

**ROC**

| Dataset  | Baseline | Random | 규칙 A | 규칙 B | 규칙 C |
|----------|----------|--------|--------|--------|--------|
| MSL      | 0.797    | 0.713  | 0.708  | 0.713  | 0.716  |
| SMAP     | 0.682    | 0.715  | 0.694  | 0.683  | 0.701  |
| SMD      | 0.949    | 0.950  | 0.949  | -      | 0.951  |

**PRC**

| Dataset  | Baseline | Random | 규칙 A | 규칙 B | 규칙 C |
|----------|----------|--------|--------|--------|--------|
| MSL      | 0.184    | 0.186  | 0.179  | 0.187  | 0.187  |
| SMAP     | 0.388    | 0.367  | 0.325  | 0.297  | 0.385  |
| SMD      | 0.666    | 0.667  | 0.659  | -      | 0.667  |

- 규칙 간 일관된 우위 없음 — 데이터셋에 따라 Random 선택이 더 높은 성능을 보이는 경우 존재
- 파생변수 추가 전후 성능 차이가 유의미한 수준에 도달하지 못함 — 다중공선성 문제로 효과 희석 추정

### 2. PCA 실험

PCA로 원 데이터의 차원을 축소한 뒤 규칙별 상호작용 항을 추가하여 성능 변화를 확인하였다.

**ROC**

| Dataset | 논문    | Baseline | PCA만 | 규칙1 | 규칙2 | 규칙3 | 규칙4 |
|---------|---------|----------|-------|-------|-------|-------|-------|
| MSL     | 0.766   | 0.797    | 0.776 | 0.764 | 0.764 | 0.708 | 0.705 |
| SMAP    | 0.727   | 0.682    | 0.707 | 0.681 | 0.652 | 0.722 | 0.350 |
| SMD     | 0.953   | 0.949    | 0.958 | 0.951 | 0.960 | 0.950 | 0.705 |

**PRC**

| Dataset | 논문    | Baseline | PCA만 | 규칙1 | 규칙2 | 규칙3 | 규칙4 |
|---------|---------|----------|-------|-------|-------|-------|-------|
| MSL     | 0.368   | 0.184    | 0.579 | 0.452 | 0.605 | 0.207 | 0.209 |
| SMAP    | 0.455   | 0.388    | 0.347 | 0.483 | 0.478 | 0.349 | 0.326 |
| SMD     | 0.750   | 0.666    | 0.786 | 0.764 | 0.765 | 0.663 | 0.209 |

- PCA 적용 시 모든 데이터셋에서 PRC 값이 논문 수치를 초과하는 경우 관찰됨
- ROC가 개선되면 PRC가 하락하는 trade-off 패턴이 반복적으로 나타남
- PCA 이후 개별 센서의 물리적 의미가 주성분에 혼합되어 해석 가능성 상실
- 원변수 간 다변수 관계 포착이라는 연구 본래 목표에서 이탈하는 방향으로 판단하여 PCA 접근 중단
  
---

## 디렉토리 구조

```
.
├── datasets/
│   ├── smd/machine-1-1/
│   ├── smap/T-3/
│   └── msl/P-15/
├── models/
│   ├── tsmixer.py     # TSMixerRevIN 모델 (gradient generator)
│   └── common.py      # ResBlock, RevIN
├── utils/
│   ├── dataloader.py  # smdDataLoader_AD (SMD / SMAP / MSL / PSM)
│   └── general.py     # set_seed
├── main.py            # 학습 루프, 10회 반복 실험, seed 관리
├── test.py            # causal matrix 계산, 이상 점수 산출, 평가 지표
├── make_runsh.py
├── 
└── run.sh             # 데이터셋별 실행 커맨드

├── datasets/
│   ├── smd/machine-1-1/
│   ├── smap/T-3/
│   └── msl/P-15/
├── models/
│   ├── tsmixer.py    
│   └── common.py     
├── utils/
│   ├── dataloader.py  
│   └── general.py  
├── main.py                    
├── test.py                    
├── causal_topn.py             # 인과 행렬 기반 상위 N 변수 쌍 선택
├── make_runsh.py              # 상호작용 항 추가 실행 커맨드 자동 생성 
├── run.sh                     # 기본 데이터셋별 실행 커맨드
├── run_interaction_rule3.sh   # 규칙 C 기반 파생변수 실험 실행 커맨드
├── run_pca.sh                 # PCA 실험 실행 커맨드
└── requirement.txt
```

---

## 데이터셋 준비

| Dataset | 설명 | 출처 |
|---------|------|------|
| SMD | 대형 인터넷 기업 서버 머신 데이터 (38 sensors, 5주) | [NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) |
| SMAP | NASA 토양수분 위성 텔레메트리 데이터 | [khundman/telemanom](https://github.com/khundman/telemanom) |
| MSL | NASA 화성 탐사선 텔레메트리 데이터 | [khundman/telemanom](https://github.com/khundman/telemanom) |

---

## 참고 논문

```
Zehao Liu, Mengzhou Gao, Pengfei Jiao.
"GCAD: Anomaly Detection in Multivariate Time Series from the Perspective of Granger Causality."
AAAI 2025. https://doi.org/10.1609/aaai.v39i18.34096
```
