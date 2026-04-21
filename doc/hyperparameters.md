# 실험 Hyperparameter 설정

## 기준점: PR #3612 리뷰어(Tom Aarsen) 실험

모든 실험은 아래 두 설정을 anchor로 삼는다.

| 항목 | Config A: GradCache Baseline | Config C: ContAccum (PR #3612) |
|------|-------------------------------|-------------------------------|
| **Model** | `microsoft/mpnet-base` | `microsoft/mpnet-base` |
| **Dataset** | `sentence-transformers/gooaq` | `sentence-transformers/gooaq` |
| **Train samples** | 100,000 | 100,000 |
| **Eval samples** | 10,000 | 10,000 |
| `per_device_train_batch_size` | 1,024 | 128 |
| `mini_batch_size` | 64 | 32 |
| `gradient_accumulation_steps` | 1 | 8 |
| **Effective batch size** | 1,024 | 1,024 |
| `learning_rate` | `8e-5` | `4e-5` |
| `warmup_ratio` | 0.1 | 0.1 |
| `num_train_epochs` | 1 | 1 |
| `cache_size` (bank) | — | 1,028 |
| `prev_cache` | — | False |
| **Training time** | 0.27 h | 0.28 h |
| **NDCG@10** | **0.8652** | **0.8304** |
| HF Hub | `tomaarsen/mpnet-base-gooaq-cmnrl-1024bs-GradCache` | `tomaarsen/mpnet-base-gooaq-cmnrl-128bs-ContAccum` |

> **왜 Config C의 NDCG@10이 낮은가?** Tom의 진단:  
> `prev_cache=False` → optimizer step마다 cache 초기화 → 실제로 bank에 쌓이는 것이 거의 없음.  
> 우리 구현은 pre-allocated circular buffer로 이 문제를 해결한다. 이것이 검증의 핵심이다.

---

## 고정 Hyperparameter (전 실험 공통)

| 항목 | 값 | 비고 |
|------|----|------|
| **Model** | `microsoft/mpnet-base` | PR과 동일 |
| **Dataset** | `sentence-transformers/gooaq` | PR과 동일 |
| **Train / Eval samples** | 100,000 / 10,000 | PR과 동일 |
| `num_train_epochs` | 1 | PR과 동일 |
| `warmup_ratio` | 0.1 | PR과 동일 |
| **Evaluation metric** | NDCG@10 | PR과 동일 |
| **Random seed** | 42 | 재현성 |

---

## 실험별 가변 Hyperparameter

### Exp-0 — 수치 동치 확인 (unit test)

훈련 없음. 동일 배치를 두 loss에 한 step 통과시켜 수치 비교.

| 항목 | 값 |
|------|----|
| `per_device_train_batch_size` | 32 |
| `mini_batch_size` | 32 |
| `query_bank_size` / `doc_bank_size` | 0 (비활성) |
| `warmup_steps` (ContAccum) | 999999 (사실상 무한) |
| **합격 기준** | `torch.allclose(loss_contaccum, loss_gradcache, atol=1e-5)` |

---

### Exp-1 — GradCache Baseline 재현

PR Config A를 그대로 재현. **목표: NDCG@10 ≈ 0.8652**.

| 항목 | 값 |
|------|----|
| **Loss** | `CachedMultipleNegativesRankingLoss` |
| `per_device_train_batch_size` | 1,024 |
| `mini_batch_size` | 64 |
| `gradient_accumulation_steps` | 1 |
| `learning_rate` | `8e-5` |

---

### Exp-2 — ContAccum Bank Size 스윕

PR Config C의 문제(bank가 사실상 비어 있음)를 우리 구현이 해결하는지 확인.  
`batch_size=128`, `grad_accum=8`로 고정하고 bank size만 변경.

| 실험 ID | `query_bank_size` | `doc_bank_size` | `warmup_steps` |
|---------|-------------------|-----------------|----------------|
| E2-a (bank 없음, PR Config C 재현) | 0 | 0 | — |
| E2-b | 256 | 256 | 50 |
| E2-c | 1,024 | 1,024 | 50 |
| E2-d | 4,096 | 4,096 | 50 |
| E2-e | 16,384 | 16,384 | 50 |

공통 고정값:

| 항목 | 값 |
|------|----|
| **Loss** | `ContrastiveAccumulationLoss` |
| `per_device_train_batch_size` | 128 |
| `mini_batch_size` | 32 |
| `gradient_accumulation_steps` | 8 |
| `learning_rate` | `4e-5` |

> E2-a는 사실상 PR Config C와 동일하므로 NDCG@10 ≈ 0.8304가 나와야 한다.  
> E2-c~e에서 0.8652를 향해 올라가는지 확인하는 것이 이 실험의 핵심이다.

---

### Exp-3 — Warmup Steps 민감도

bank가 충분히 채워지기 전 학습 불안정 여부 확인.

| 실험 ID | `warmup_steps` |
|---------|---------------|
| E3-a | 0 |
| E3-b | 50 |
| E3-c | 200 |
| E3-d | 500 |

공통 고정값:

| 항목 | 값 |
|------|----|
| `per_device_train_batch_size` | 128 |
| `mini_batch_size` | 32 |
| `gradient_accumulation_steps` | 8 |
| `learning_rate` | `4e-5` |
| `query_bank_size` / `doc_bank_size` | 1,024 |

---

### Exp-4 — PR #3612 성능 하락 원인 규명

Tom의 진단("cache gets reset with only 1 element")을 직접 재현하고, 우리 구현이 이를 해결하는지 비교.

| 실험 ID | 설명 | `on_optimizer_step` 동작 |
|---------|------|--------------------------|
| E4-a | **우리 구현** (지속 누적) | `_train_step` 증가만, bank 유지 |
| E4-b | **PR 방식 시뮬레이션** (step마다 bank 초기화) | `_train_step` 증가 + bank count/ptr 초기화 |

공통 고정값:

| 항목 | 값 |
|------|----|
| `per_device_train_batch_size` | 128 |
| `mini_batch_size` | 32 |
| `gradient_accumulation_steps` | 8 |
| `learning_rate` | `4e-5` |
| `query_bank_size` / `doc_bank_size` | 1,024 |
| `warmup_steps` | 50 |

> E4-b의 결과가 PR Config C (0.8304)와 유사하게 나오면 원인 확정.

---

### Exp-5 — Ablation: Hard Negative를 Bank에 포함 vs 제외

우리 구현은 `reps[1:]` 전체(positive + hard negative)를 bank에 저장한다.  
positive만 저장하는 경우와 비교.

| 실험 ID | bank 구성 |
|---------|----------|
| E5-a | positive + hard negative (우리 기본값) |
| E5-b | positive만 저장 |

공통 고정값: Exp-2 E2-c와 동일 (`bank=1,024`, `grad_accum=8`, `lr=4e-5`).

---

## 요약 테이블

| 실험 | 핵심 가변 항목 | 기대 결과 |
|------|---------------|----------|
| Exp-0 | — | 두 loss 수치 일치 |
| Exp-1 | baseline 재현 | NDCG@10 ≈ 0.8652 |
| Exp-2 | bank size 0→16384 | 0.8304 → 0.8652 방향으로 회복 |
| Exp-3 | warmup_steps 0→500 | 너무 낮으면 초기 불안정 |
| Exp-4 | bank reset 정책 | E4-b ≈ 0.8304, E4-a > 0.8304 |
| Exp-5 | bank 구성 | hard neg 포함이 더 낫거나 비슷 |
