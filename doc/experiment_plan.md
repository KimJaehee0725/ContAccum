# ContrastiveAccumulationLoss 검증 실험 계획

## 배경

이 계획은 [sentence-transformers PR #3612](https://github.com/huggingface/sentence-transformers/pull/3612)의 검토 결과를 참고하여 작성되었다.
PR에서 리뷰어(Tom Aarsen)는 자체 ContAccum 구현이 GradCache 단독 대비 **NDCG@10 기준 성능 하락**(0.8652 → 0.8304)을 보였다고 보고했다.
우리의 구현이 이 문제를 재현하는지, 아니면 극복하는지를 체계적으로 검증하는 것이 목표다.

---

## 핵심 질문

1. **정확성**: 우리 구현이 동일 설정(batch size, bank 비활성화)에서 GradCache와 수치적으로 동일한 loss를 내는가?
2. **기대 효과**: bank를 사용하면 낮은 메모리(small batch)에서 큰 배치 효과를 재현할 수 있는가?
3. **PR 재현**: PR #3612가 보고한 성능 하락이 구현 결함인지, ContAccum 자체의 한계인지를 판별할 수 있는가?
4. **하이퍼파라미터 민감도**: `bank_size`와 `warmup_steps`가 성능에 어떤 영향을 주는가?

---

## 실험 설계

### Exp-0. 수치 동치 확인 (Unit Test, 코드 실행 전 필수)

**목적**: bank를 끈 상태(`query_bank_size=0` 또는 `warmup_steps=∞`)에서 `ContrastiveAccumulationLoss`와 `CachedMultipleNegativesRankingLoss`의 loss 값이 일치하는지 확인.

**방법**:
- 동일 시드, 동일 배치로 두 loss 클래스 각각 1 step 실행
- `torch.allclose(loss_contaccum, loss_cached_mnrl, atol=1e-5)` 통과 여부 기록

**합격 기준**: 두 loss가 `atol=1e-5` 이내에서 일치

---

### Exp-1. GradCache Baseline 재현

**목적**: PR #3612 리뷰어의 baseline 수치(NDCG@10 = 0.8652)를 우리 환경에서 재현하여 비교 기준점 확보.

| 항목 | 설정 |
|------|------|
| 모델 | `microsoft/mpnet-base` (PR과 동일 계열) |
| Loss | `CachedMultipleNegativesRankingLoss` |
| 데이터 | MS MARCO (또는 NQ) — PR에서 사용한 데이터셋 |
| `per_device_train_batch_size` | 384 (high-resource baseline) |
| `mini_batch_size` | 32 |
| Metric | NDCG@10 on BEIR (or MS MARCO dev) |

**기록 항목**: loss curve, NDCG@10, 훈련 시간, GPU peak memory

---

### Exp-2. ContAccum — bank size 스윕

**목적**: bank size가 커질수록 성능이 올라가는지 확인. 논문의 핵심 주장 검증.

| 실험 ID | `per_device_train_batch_size` | `query_bank_size` = `doc_bank_size` | `warmup_steps` |
|---------|-------------------------------|-------------------------------------|----------------|
| E2-a    | 32 (low memory)               | 0 (bank 없음)                       | —              |
| E2-b    | 32                            | 512                                 | 50             |
| E2-c    | 32                            | 2048                                | 50             |
| E2-d    | 32                            | 8192                                | 50             |
| E2-e    | 32                            | 65536                               | 50             |
| E2-ref  | 384 (high resource)           | 0                                   | —              |

**기대 결과**: E2-e가 E2-ref에 근접해야 논문 주장이 맞다.
**확인 질문**: 어느 bank size부터 E2-ref 대비 gap이 메워지기 시작하는가?

---

### Exp-3. warmup_steps 민감도

**목적**: 초기 step에서 bank가 비어 있을 때 부실한 representation으로 인한 학습 불안정 여부 파악.

| 실험 ID | `warmup_steps` | 비고 |
|---------|---------------|------|
| E3-a    | 0             | 즉시 bank 사용 |
| E3-b    | 50            |  |
| E3-c    | 200           |  |
| E3-d    | 500           |  |

**고정 조건**: `batch_size=32`, `bank_size=8192`
**기록**: loss curve의 초기 변동성(std of loss at step 1–100), 최종 NDCG@10

---

### Exp-4. PR #3612 재현 실험 — 성능 하락 원인 규명

**목적**: PR 리뷰어가 보고한 성능 하락(0.8652 → 0.8304)이 구현 방식의 차이에서 비롯된 것인지 진단.

PR #3612와 우리 구현의 주요 차이점:

| 항목 | PR #3612 구현 | 우리 구현 |
|------|--------------|----------|
| bank 자료구조 | `deque` (Python) | pre-allocated tensor (FIFO) |
| hard negative 처리 | 미확인 | `reps[1:]` 전체 저장 |
| bank 초기화 시점 | 명시적 `reset_cont_cache()` | lazy init (first forward) |
| Trainer 연동 | `_LossCacheResetCallback` | 없음 (loss 자체가 관리) |
| gradient accumulation 상호작용 | `on_optimizer_step()` 로 step마다 cache reset 가능 | 없음 |

**실험**:
- E4-a: PR #3612 방식처럼 optimizer step마다 bank를 reset하는 경우 vs. 지속 누적하는 경우 비교
- E4-b: hard negative를 bank에서 제외했을 때(positive만 저장) vs. 포함했을 때 비교
- E4-c: `prev_cache=False` (step마다 초기화) vs. `prev_cache=True` (누적) 비교

---

### Exp-5. 논문 수치 재현

**목적**: NeurIPS 2024 논문 Table의 핵심 결과를 sentence-transformers 환경에서 재현.

| 논문 결과 | 설정 | 목표 metric |
|-----------|------|------------|
| ContAccum (11GB) ≥ DPR (80GB) | MS MARCO, NQ | MRR@10, R@k |
| ContAccum > GradCache (같은 메모리) | 5 IR 벤치마크 | NDCG@10 |

**데이터셋**: MS MARCO Passage Ranking, Natural Questions, TriviaQA, TREC, SQuAD (논문과 동일)
**평가 도구**: `beir` 라이브러리 또는 `sentence-transformers` 내장 evaluator

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| NDCG@10 | 검색 랭킹 품질 (primary) |
| MRR@10 | Mean Reciprocal Rank |
| Recall@{10, 100} | 후보 recall |
| GPU peak memory (GB) | `torch.cuda.max_memory_allocated()` |
| Steps/sec | 훈련 처리량 |
| Loss std (step 1–100) | 초기 학습 안정성 |

---

## 실험 순서 (우선순위)

```
Exp-0 (수치 동치 확인)
  └─ Exp-1 (GradCache baseline)
       ├─ Exp-2 (bank size 스윕)  ← 핵심
       ├─ Exp-3 (warmup 민감도)
       ├─ Exp-4 (PR 하락 원인 규명)
       └─ Exp-5 (논문 재현)       ← 최종 목표
```

Exp-0은 코드 변경 시마다 반복 실행. Exp-1 없이 Exp-2 이후를 진행하면 비교 기준이 없으므로 반드시 선행.

---

## 실험 로깅 체크리스트

각 실험 실행 시 반드시 기록:
- [ ] 커밋 해시 (`git rev-parse HEAD`)
- [ ] 환경 (`uv pip freeze`)
- [ ] 시드 (`--seed`)
- [ ] 전체 훈련 설정 (`TrainingArguments` dump)
- [ ] Eval 결과 전체 (단일 숫자 아닌 전 metric)
- [ ] GPU peak memory
- [ ] 훈련 시간

---

## 참고

- PR #3612: https://github.com/huggingface/sentence-transformers/pull/3612
- ContAccum 논문: https://arxiv.org/abs/2406.12356
- GradCache 논문: https://huggingface.co/papers/2101.06983
