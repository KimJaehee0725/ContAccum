# Contrastive Accumulation (ContAccum)

## 논문 정보

| 항목 | 내용 |
|------|------|
| **제목** | A Gradient Accumulation Method for Dense Retriever under Memory Constraint |
| **저자** | Jaehee Kim, Yukyung Lee, Pilsung Kang |
| **소속** | Seoul National University |
| **학회** | NeurIPS 2024 (38th Conference on Neural Information Processing Systems) |
| **날짜** | 2024년 12월 10일 |
| **arXiv** | [arxiv.org/abs/2406.12356](https://arxiv.org/abs/2406.12356) |
| **NeurIPS** | [proceedings link](https://proceedings.neurips.cc/paper_files/paper/2024/file/15ba84c1e19b0eb75f96922f5da0a021-Paper-Conference.pdf) |
| **공식 코드** | [DSBA-Lab/Contrastive-Accumulation](https://github.com/DSBA-Lab/Contrastive-Accumulation) |

---

## Abstract

> InfoNCE loss is commonly used to train dense retriever in information retrieval tasks. It is well known that a large batch is essential to stable and effective training with InfoNCE loss, which requires significant hardware resources. Due to the dependency of large batch, dense retriever has bottleneck of application and research. Recently, memory reduction methods have been broadly adopted to resolve the hardware bottleneck by decomposing forward and backward or using a memory bank. However, current methods still suffer from slow and unstable training. To address these issues, we propose Contrastive Accumulation (ContAccum), a stable and efficient memory reduction method for dense retriever trains that uses a dual memory bank structure to leverage previously generated query and passage representations. Experiments on widely used five information retrieval datasets indicate that ContAccum can surpass not only existing memory reduction methods but also high-resource scenario. Moreover, theoretical analysis and experimental results confirm that ContAccum provides more stable dual-encoder training than current memory bank utilization methods.

---

## 문제 정의

Dense retriever (예: DPR)는 InfoNCE loss를 사용하여 훈련되며, 이 loss는 **대규모 배치**를 필요로 한다.

- 많은 양의 in-batch negative sample이 있어야 효과적으로 수렴
- 대규모 배치 → 높은 GPU 메모리 요구 (예: 80GB 이상)
- 이로 인해 연구 및 실용화에 병목 발생

기존 해결책과 한계:
| 방법 | 아이디어 | 한계 |
|------|----------|------|
| GradCache | forward/backward 분리로 메모리 절감 | 훈련 속도 느림, 불안정 |
| Memory Bank (단방향) | 이전 표현을 저장해 재사용 | query 또는 passage 한쪽만 저장 → 불안정 |

---

## 제안 방법: Contrastive Accumulation (ContAccum)

### 핵심 아이디어: Dual Memory Bank

- **Query Memory Bank**와 **Passage Memory Bank** 두 개를 동시에 운용
- 이전 스텝에서 생성된 query/passage representation을 누적 저장
- 현재 배치의 negative sample pool을 크게 확장함으로써 소규모 배치에서도 InfoNCE loss를 안정적으로 최적화

### 작동 방식

1. 미니배치에서 query/passage를 인코딩
2. 현재 인코딩 결과를 두 메모리 뱅크에 enqueue
3. InfoNCE loss 계산 시, 현재 배치 + 메모리 뱅크에 저장된 과거 표현을 모두 negative로 활용
4. 오래된 표현은 dequeue (FIFO)

### 이론적 분석

- 단방향 메모리 뱅크(query 또는 passage만 저장)를 쓰면 gradient 편향이 발생
- Dual 메모리 뱅크는 query-passage 쌍의 분포를 균형 있게 유지 → **더 안정적인 학습**

---

## 실험 결과

### 평가 데이터셋

5개의 표준 정보검색 벤치마크 (MS MARCO, Natural Questions 등 포함)

### 주요 결과

| 비교 | 결과 |
|------|------|
| ContAccum (11GB) vs. DPR 고자원 (80GB) | **ContAccum 승리** — 24개 지표 중 18개에서 우위, 최대 4.9점 향상 |
| vs. GradCache 등 기존 메모리 절감법 | ContAccum이 전반적으로 우수 |
| 메모리 제약 간 일관성 (11GB vs 24GB) | 성능 차이 미미 → **안정적** |

---

## 이 레포의 목적: Sentence Transformers 이식

이 레포(`ContAccum`)는 ContAccum 알고리즘을 **[Sentence Transformers](https://www.sbert.net/)** 프레임워크에 이식하기 위한 작업 공간이다.

- 원 구현은 DPR 기반 커스텀 학습 코드
- Sentence Transformers는 범용 문장 임베딩/검색 프레임워크로 생태계가 넓음
- 이식을 통해 더 다양한 모델/데이터셋/태스크에 ContAccum을 적용 가능

### 관련 Sentence Transformers 개념

| 개념 | 설명 |
|------|------|
| `SentenceTransformer` | 문장 인코더 모델 래퍼 |
| `MultipleNegativesRankingLoss` | InfoNCE와 유사한 in-batch negative loss |
| `CachedMultipleNegativesRankingLoss` | GradCache 방식의 메모리 절감 구현체 |
| ContAccum 이식 목표 | Dual memory bank를 Sentence Transformers 훈련 루프에 통합 |

---

## 참고 자료

- [arXiv 논문](https://arxiv.org/abs/2406.12356)
- [NeurIPS 2024 포스터](https://neurips.cc/virtual/2024/poster/95253)
- [공식 GitHub](https://github.com/DSBA-Lab/Contrastive-Accumulation)
- [Sentence Transformers 공식 문서](https://www.sbert.net/)
- [sentence-transformers GitHub](https://github.com/UKPLab/sentence-transformers)
