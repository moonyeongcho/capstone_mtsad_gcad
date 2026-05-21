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

모든 실험은 10회 반복 후 평균값을 사용한다. 
10번의 반복 실험은 재현성을 위해 시드를 고정하였다.
아래 수치는 팀 자체 재현 baseline으로, 논문 원본 수치와 차이가 있다. (시드 미공개 및 코드 부분 공개)

| Dataset | ROC-AUC | PRC-AUC |
|---------|---------|---------|
| SMD (machine-1-1) | 0.9494 | 0.6656 |
| SMAP (T-3) | 0.6817 | 0.3879 |
| MSL (P-15) | 0.7973 | 0.1841 |

---

## 데이터셋 준비

| Dataset | 설명 | 출처 |
|---------|------|------|
| SMD | 대형 인터넷 기업 서버 머신 데이터 (38 sensors, 5주) | [NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) |
| SMAP | NASA 토양수분 위성 텔레메트리 데이터 | [khundman/telemanom](https://github.com/khundman/telemanom) |
| MSL | NASA 화성 탐사선 텔레메트리 데이터 | [khundman/telemanom](https://github.com/khundman/telemanom) |

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
└── run.sh             # 데이터셋별 실행 커맨드
```

---

## 참고 논문

```
Zehao Liu, Mengzhou Gao, Pengfei Jiao.
"GCAD: Anomaly Detection in Multivariate Time Series from the Perspective of Granger Causality."
AAAI 2025. https://doi.org/10.1609/aaai.v39i18.34096
```
