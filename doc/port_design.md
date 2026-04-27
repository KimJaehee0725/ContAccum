# ContrastiveAccumulationLoss 포트 설계 문서

> 대상 업스트림: `huggingface/sentence-transformers` main (5.5.0.dev0, 2026-04-22 기준)
> 작성 근거: PR #3612 diff, 현재 main `cached_multiple_negatives_ranking.py`, `trainer.py`, paper summary, experiment plan, hyperparameters doc

---

## 1. Class surface

현재 main의 canonical 레이아웃은 예전 `sentence_transformers/losses/`가 아니라 중첩된 `sentence_transformers/sentence_transformer/losses/`다. 따라서 새 loss의 구현 위치는 다음으로 둔다.

`sentence_transformers/sentence_transformer/losses/contrastive_accumulation.py`

정확한 생성자 시그니처는, 현재 `CachedMultipleNegativesRankingLoss` public surface를 유지하면서 ContAccum-specific 인자만 추가하는 쪽으로 고정한다.

```python
def __init__(
    self,
    model: SentenceTransformer,
    scale: float = 20.0,
    similarity_fct: Callable[[Tensor, Tensor], Tensor] = util.cos_sim,
    mini_batch_size: int = 32,
    bank_size: int = 0,
    warmup_steps: int = 0,
    gather_across_devices: bool = False,
    directions: tuple[Literal["query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"], ...] = ("query_to_doc",),
    partition_mode: Literal["joint", "per_direction"] = "joint",
    show_progress_bar: bool = False,
    hardness_mode: Literal["in_batch_negatives", "hard_negatives", "all_negatives"] | None = None,
    hardness_strength: float = 0.0,
) -> None
```

근거:

- 현재 `sbert.net`의 `CachedMultipleNegativesRankingLoss` docs는 이미 `directions`, `partition_mode`, `hardness_mode`, `hardness_strength`를 public surface로 노출한다.
- `mini_batch_size`와 2-pass GradCache flow는 현재 `cached_multiple_negatives_ranking.py`의 핵심 surface다.
- `bank_size`는 ContAccum dual bank의 sample capacity이고, `bank_size=0`이면 기능이 완전히 꺼진다.
- `warmup_steps`는 optimizer-step 기준 bank activation gate다.

`forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor | None = None) -> Tensor` 의미는 다음과 같이 둔다.

1. current batch만 현재 `CachedMultipleNegativesRankingLoss`와 동일하게 `mini_batch_size` 단위 no-grad 1차 embedding 한다.
2. 그 결과로 얻은 current-step `query_reps`와 `passage_reps = reps[1:]`를 detached leaf로 만들고, bank가 활성화된 경우 paper Eq. 5-7의 형태로 `Q = [Q_cur; sg(Q_bank)]`, `P = [P_cur; sg(P_bank)]`를 구성한다.
3. 여기서 `P_cur`와 `P_bank`는 `reps[1:]` 전체를 보존한다. 즉 `reps[1]`의 positive만이 아니라 `reps[2:]` 이후 explicit hard negatives까지 모두 포함한다.
4. loss-to-embedding gradient cache와 `_backward_hook` 기반 re-embed는 current-step tensor에만 적용한다. bank tensor는 second pass에 다시 들어오지 않는다.
5. same optimizer step 안의 micro-batch는 서로를 bank negative로 보지 않는다. current-step에서 얻은 detached embedding은 pending staging에만 넣고, committed bank는 optimizer step 후 `on_optimizer_step()`에서만 전진시킨다.
6. `labels`는 현재 MNRL 계열과 동일하게 실질적으로 사용하지 않는다.

export는 두 단계로 둔다.

- canonical export: `sentence_transformers/sentence_transformer/losses/__init__.py`에 `ContrastiveAccumulationLoss` 추가
- deprecated import shim: 루트 `sentence_transformers/__init__.py`가 이미 `setup_deprecated_module_imports()`를 호출하므로, assumption: deprecated import map에 `sentence_transformers.losses.ContrastiveAccumulationLoss`를 추가해 old import를 유지한다

---

## 2. Bank data structure

paper summary와 NeurIPS PDF Eq. 5-7은 query bank `M_q`와 passage bank `M_p`의 dual bank를 전제로 한다. v1 구현도 이 구조를 그대로 따른다. PR #3612의 `deque` 기반 cache는 사용하지 않는다.

committed bank는 pre-allocated circular tensor buffer 두 개로 둔다.

- `query_bank`: shape `(bank_size, embedding_dim)`
- `passage_bank`: shape `(bank_size, num_passage_columns, embedding_dim)`

여기서 `num_passage_columns = len(reps) - 1`이다.

- `passage_bank[:, 0, :]`는 positive column
- `passage_bank[:, 1:, :]`는 explicit hard negatives (`reps[2:]` 이후)
- pair training이면 `num_passage_columns == 1`
- triplet / n-tuple training이면 같은 slot 안에 positive와 hard negatives를 함께 저장한다

allocation 정책:

- first train forward에서 `embedding_dim`, `dtype`, `device`, `num_passage_columns`를 보고 lazy allocation 한다
- 이후 `num_passage_columns`가 달라지면 `ValueError`로 막는다
- assumption: 한 loss instance가 소비하는 dataset schema는 고정이다

detach / grad 정책:

- bank에 저장되는 tensor는 항상 `.detach()` 상태이며 `requires_grad=False`
- current-step transient tensor만 현재 cached loss와 동일하게 `detach().requires_grad_()` leaf로 바꿔 gradient cache를 계산한다
- bank에서 읽어온 tensor는 forward 내내 stop-gradient constant다

device / dtype:

- bank tensor는 first train forward의 embedding output과 같은 `device` / `dtype`로 생성한다
- v1은 CPU offload를 하지 않는다. similarity matrix를 step마다 바로 계산해야 하므로 same-device 유지가 가장 단순하고 현재 cached loss와도 맞는다

write tracking은 Python container가 아니라 int scalar로 유지한다.

- `self._bank_ptr: int`
- `self._bank_fill: int`
- `self._optimizer_steps: int`

step-local staging은 optimizer-step commit 전까지 따로 둔다.

- `self._pending_queries: list[Tensor]`
- `self._pending_passages: list[Tensor]`

이 staging list는 gradient accumulation 동안 여러 forward에서 나온 current-step detached embeddings를 모으는 용도다. 실제 circular bank overwrite는 optimizer step 후 한 번만 일어난다.

warmup / partial fill / eval 동작:

- `bank_size == 0`: allocation, read, write, callback 전부 no-op
- `0 < _bank_fill < bank_size`: committed bank의 `[:_bank_fill]` prefix만 negative pool로 읽는다
- `_optimizer_steps < warmup_steps`: bank read/write 둘 다 skip한다
- `model.training is False` 또는 `torch.is_grad_enabled() is False`: bank read/write 둘 다 no-op이다. eval loss는 current batch만 본다
- pending staging도 eval에서는 쌓지 않는다

---

## 3. Trainer integration

bank write pointer는 forward가 아니라 optimizer step 직후 전진시킨다. loss 쪽 API는 다음 한 개로 고정한다.

```python
def on_optimizer_step(self) -> None:
    ...
```

이 메서드는 다음을 수행한다.

1. `_pending_queries`와 `_pending_passages`를 time order대로 concat한다
2. circular overwrite 규칙으로 `query_bank` / `passage_bank`에 commit한다
3. `_bank_ptr`, `_bank_fill`, `_optimizer_steps`를 갱신한다
4. pending staging을 clear한다

선택은 internal trainer hook이 아니라 `TrainerCallback` subclass다.

이유:

- 현재 main의 `sentence_transformers/sentence_transformer/trainer.py`는 Hugging Face trainer 위에 얇게 얹힌 wrapper이고, constructor surface에 이미 `callbacks`를 노출한다
- optimizer loop 자체를 fork하지 않고도 loss side-effect를 삽입할 수 있다
- PR #3612도 같은 문제를 callback으로 풀었지만, 그 callback은 reset semantics를 넣어서 shape가 잘못됐다. callback 자체의 방향은 유지하되 책임을 "reset"이 아니라 "commit"으로 바꿔야 한다

권장 형태:

- `ContrastiveAccumulationCommitCallback(TrainerCallback)`
- callback event는 `on_optimizer_step`를 사용한다
- trainer가 가진 active loss object를 순회하고, `hasattr(loss_fn, "on_optimizer_step")`인 경우에만 호출한다
- loss dict를 받는 current trainer usage를 고려해, dict value들도 함께 순회한다

어느 trainer가 hook을 가져야 하는가:

- `SentenceTransformerTrainer`: Yes. `ContrastiveAccumulationLoss`의 실제 사용처이고, 현재 사용자가 읽으라고 지정한 main trainer도 이 파일이다.
- `CrossEncoderTrainer`: No. ContAccum은 dual-encoder embedding loss이고, cross-encoder의 logit cache path와는 수학도 코드 경로도 다르다.
- `SparseEncoderTrainer`: No for v1. sparse encoder는 별도 loss namespace와 scoring path를 가지며, 본 설계의 bank 구조는 dense sentence embedding tensor를 전제로 한다.

즉 auto-registration은 `SentenceTransformerTrainer`에만 넣고, 다른 trainer까지 공용 hook으로 넓히지 않는 것이 최소 변경 범위다.

---

## 4. Gradient correctness

핵심은 "current-step sample에 대한 gradient만" larger physical batch와 동일하다는 점이다.

현재 `cached_multiple_negatives_ranking.py`는 이미 GradCache의 표준 2-pass 수학을 쓴다.

1. no-grad 1차 embedding
2. embedding-level loss 계산 후, embedding에 대한 gradient cache
3. `loss.register_hook(...)`로 current-step sample만 다시 embed

ContAccum에서 bank tensor를 붙이는 방식도 이 구조를 그대로 따른다. paper Eq. 5-7은 `Q = [Q_cur; sg(M_q)]`, `P = [P_cur; sg(M_p)]`를 정의하므로, bank sample은 stop-gradient constant로 loss에 들어간다.

따라서 detached bank를 추가한 후의 loss는, current-step embedding에 대해서는 "더 큰 similarity matrix를 사용한 loss"의 partial derivative와 일치한다.

- current query는 bank passage를 extra negatives로 본다
- current passage는 bank query row들로부터 additional contrastive pressure를 받는다
- same-step current samples끼리의 gradient path는 current cached loss의 second pass로 정확히 복원된다

하지만 true larger physical batch와 완전히 같다고 말하면 안 된다.

- bank sample 자체는 detached라서 그 sample의 parameter gradient는 0이다
- true larger batch라면 그 sample들도 current parameter로 다시 embed되고 자기 own loss row/column을 통해 gradient를 받는다
- 따라서 동일성은 "현재 step에서 다시 embed된 sample들의 gradient" 범위로만 성립한다

이 가정이 깨지는 지점은 정확히 세 곳이다.

1. `all_gather_with_grad`
   현재 cached loss는 multi-GPU current batch를 `all_gather_with_grad`로 모아 gradient-preserving global batch를 만든다. detached bank tensor는 이 경로 밖에 있다. bank까지 global하게 맞추려면 rank-synchronized committed bank와 no-grad gather를 별도로 설계해야 한다.

2. multi-GPU semantics
   rank-local bank만 두면 rank마다 negative pool이 달라진다. global bank를 만들면 commit order, uneven last batch, label offset을 모두 동기화해야 한다. v1에서는 `bank_size > 0 and gather_across_devices=True`를 hard error로 두는 것이 가장 안전하다.

3. temperature / scale interaction
   현재 MNRL 계열은 similarity에 `scale = 1 / temperature`를 곱한 뒤 softmax/logsumexp를 계산한다. detached stale negatives도 denominator에 그대로 들어가므로 current-step gradient magnitude는 바뀐다. larger physical batch도 denominator effect는 비슷하지만, 그쪽은 stale bank가 아니라 current-parameter embedding이고 banked sample 자체 gradient도 존재한다. 따라서 "완전 동일"이 아니라 "current-step local gradient만 동일"이라고 서술해야 한다.

같은 caveat는 `hardness_mode`가 켜진 경우에도 유지된다. weighting에 쓰이는 similarity는 stop-grad일 수 있어도, bank sample 자체에 gradient가 생기지는 않는다.

---

## 5. Exp-0 gate

Exp-0 gate는 bank가 완전히 꺼졌을 때 `ContrastiveAccumulationLoss`가 current `CachedMultipleNegativesRankingLoss`와 수치적으로 같음을 확인하는 단일 unit test로 둔다.

정확한 테스트 위치와 node id:

- file path: `tests/sentence_transformer/losses/test_contrastive_accumulation_loss.py`
- pytest node id: `tests/sentence_transformer/losses/test_contrastive_accumulation_loss.py::test_bank_disabled_matches_cached_multiple_negatives_ranking_loss`

path 선택 근거는 현재 main의 `tests/` 레이아웃이 model family 별로 분리되어 있고 (`tests/sentence_transformer/`, `tests/cross_encoder/`, `tests/sparse_encoder/`, `tests/base/`), dense sentence-encoder loss 테스트는 이미 `tests/sentence_transformer/losses/` 아래에 놓인다는 점이다 (예: `tests/sentence_transformer/losses/test_cmnrl.py`). 새 dense loss test도 같은 디렉터리에 추가한다.

fixture는 최소 세 개로 충분하다.

- `model`: deterministic tiny `SentenceTransformer`
- `sentence_features`: 하나의 고정 batch. 최소 `(anchor, positive)` pair batch, 가능하면 `(anchor, positive, hard_negative)` variant도 같은 파일에 추가
- `labels`: `torch.empty(0)`

테스트 절차:

1. `torch.manual_seed(42)`로 seed 고정
2. 동일한 초기 weight의 model clone 두 개 준비
3. `cached_loss = CachedMultipleNegativesRankingLoss(model_a, mini_batch_size=..., gather_across_devices=False, ...)`
4. `contaccum_loss = ContrastiveAccumulationLoss(model_b, mini_batch_size=..., bank_size=0, warmup_steps=0, gather_across_devices=False, ...)`
5. 동일 `sentence_features`를 넣어 scalar loss를 계산
6. `torch.allclose(loss_contaccum, loss_cached, atol=1e-5)` assert

baseline disable switch는 `bank_size=0`으로 고정한다.

- `bank_size=0`은 bank allocation, bank read, bank write, callback commit, warmup state machine 전부를 제거한다
- `warmup_steps=inf`는 기능을 "사용 안 함"이 아니라 "stateful but never activated"로 남긴다
- Exp-0의 목적은 feature-off baseline이 `CachedMultipleNegativesRankingLoss`와 정확히 같은지를 보는 것이므로 `bank_size=0`가 더 강한 gate다

이 테스트가 실제로 assert하는 것은 딱 하나다.

- identical batch + identical seed + bank disabled 조건에서 `ContrastiveAccumulationLoss(..., bank_size=0)`의 loss tensor가 `CachedMultipleNegativesRankingLoss`의 loss tensor와 `atol=1e-5` 이내로 동일하다

---

## 6. Migration from pr-3612

| pr-3612 construct | New design | Replacement / drop reason |
|---|---|---|
| `use_cont_accum` | Drop | standalone class 선택 자체가 feature flag다 |
| `cache_size` | Replace with `bank_size` | micro-batch cache가 아니라 sample-slot 기준 circular bank capacity를 뜻하도록 이름을 명확화한다 |
| `prev_cache` | Drop | 학습 중 bank를 step마다 비우는 옵션 자체가 ContAccum 목적과 충돌한다 |
| `prev_cache=False` path | Remove entirely | Tom Aarsen이 지적한 regression의 직접 원인이다 |
| `_query_cache: deque` | Replace | `query_bank` pre-allocated tensor + `_bank_ptr` / `_bank_fill` |
| `_candidate_caches: list[deque]` | Replace | `passage_bank` pre-allocated tensor; hard negatives는 2nd axis에 함께 저장 |
| `_collect_cont_cache` | Replace | deque를 매 step `torch.cat`하는 대신 committed bank slice view를 바로 읽는다 |
| `_enqueue_cont_cache` | Replace | forward-time immediate enqueue 대신 pending staging 후 optimizer-step commit |
| `reset_cont_cache()` | Drop from normal path | debug helper면 몰라도 training path에서는 자동 reset 금지 |
| `_LossCacheResetCallback` | Replace | `ContrastiveAccumulationCommitCallback`; reset이 아니라 commit만 수행 |
| trainer-side `_loss_on_optimizer_step` helper | Replace | generic "walk active losses and call `on_optimizer_step()` if present" helper |
| loss-side `on_optimizer_step()` | Keep, semantics change | reset이 아니라 circular bank commit와 step counter advance만 담당 |

Tom Aarsen-identified regression 회피 포인트는 명확하다.

- PR #3612는 `prev_cache=False`일 때 optimizer step마다 cache를 clear했다
- 그 결과 bank가 FIFO memory가 아니라 사실상 step-local scratch space가 됐다
- 특히 `gradient_accumulation_steps > 1` 환경에서는 previous-step negatives가 유지되지 않아 ContAccum 효과가 사라질 수 있다
- 새 설계는 `prev_cache` 자체를 없애고, bank를 eviction-only circular FIFO로 고정한다
- 따라서 optimizer step은 bank를 비우지 않고, 오직 pending sample을 append/overwrite만 한다

---

## 7. Open questions — 저자 결정 (2026-04-22)

1. **`bank_size` API** → **단일 `bank_size`** 확정. `query_bank_size` / `passage_bank_size` 분리는 하지 않는다. 향후 실험에서 개별 튜닝이 필요해지면 별도 PR로 확장.

2. **`gather_across_devices=True` + `bank_size > 0`** → **v1에서 hard error** 확정. 생성자에서 두 조건이 동시에 참이면 다음 메시지의 `ValueError`를 발생시킨다:

   > `"ContrastiveAccumulationLoss does not support gather_across_devices=True together with bank_size > 0 in v1. Multi-GPU global bank synchronization is planned as a follow-up; for now, either set bank_size=0 or run single-GPU."`

3. **`num_passage_columns` 불일치** → **`ValueError`** 확정. 에러 메시지에 첫 batch에서 관측한 column 수와 현재 batch의 column 수를 모두 포함해 디버그가 쉽도록 한다.

**발견된 contradiction:**

- 이 task 지시서에 적힌 `sentence_transformers/losses/` 및 `sentence_transformers/trainer.py` 경로는 current main과 맞지 않는다. current main의 canonical path는 `sentence_transformers/sentence_transformer/losses/`와 `sentence_transformers/sentence_transformer/trainer.py`다.
- current public docs는 여전히 old import path `sentence_transformers.losses.*`를 상당 부분 노출하지만, current code/tree는 nested `sentence_transformers.sentence_transformer.losses.*`를 canonical로 사용한다. 즉 docs와 code가 완전히 동기화되어 있지 않다.
