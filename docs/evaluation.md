# Evaluation Guide

이 문서는 현재 `data/novels` 구조 기준으로 일반 eval과 hardset 평가 실행 방법을 정리합니다.

## 1) 평가 데이터셋 종류

작품별로 아래 2종 JSONL을 사용합니다.

- 일반 평가셋: `<작품명>-eval.jsonl`
- 하드 평가셋: `<작품명>-hardset.jsonl`

예시:

- `data/novels/김유정-동백꽃-eval.jsonl`
- `data/novels/김유정-동백꽃-hardset.jsonl`

Hardset에는 `tags`로 `alias-noise`, `multi-hop`, `3-hop`/`4-hop` 등이 포함되어 있습니다.

## 2) 입력 스키마(JSONL)

각 줄은 1개 평가 예제(JSON object)입니다.

필수 필드:

- `query` (string)
- `reference_entities` (list[string])

선택 필드:

- `id`
- `reference_relationships` (`source`, `target`, `type`)
- `ground_truth`
- `tags`

## 3) 실행 전 준비

1. 그래프 생성

```bash
uv run python main.py process "data/novels/현진건-운수좋은날.txt" -o "data/novels/현진건-운수좋은날-KG.json"
```

2. 환경 변수

```bash
export OPENAI_API_KEY="your-api-key"
```

## 4) 실행 커맨드

### 일반 평가

```bash
uv run python main.py eval \
  --dataset "data/novels/현진건-운수좋은날-eval.jsonl" \
  -g "data/novels/현진건-운수좋은날-KG.json" \
  -o "data/novels/현진건-운수좋은날-eval-results.json"
```

### Hardset 평가

```bash
uv run python main.py eval \
  --dataset "data/novels/현진건-운수좋은날-hardset.jsonl" \
  -g "data/novels/현진건-운수좋은날-KG.json" \
  --hops 4 \
  -o "data/novels/현진건-운수좋은날-hardset-results.json"
```

추천:

- 일반셋 baseline: `--hops 2`
- hard multi-hop 포함 시: `--hops 3` 또는 `--hops 4`
- retrieval만 보고 싶으면 `--skip-generation` 사용

## 5) 출력 결과 파일

평가 결과 JSON(`<작품명>-*-results.json`) 구조:

- `summary`: 전체 평균 지표/지연/토큰/예상비용
- `results`: 예제별 retrieved entities + 메트릭

대표 메트릭:

- `precision_at_k`
- `recall_at_k`
- `mrr`
- `ndcg_at_k`

## 6) `data/novels` 네이밍 규칙

작품별 산출물은 아래 naming convention을 사용합니다.

- 원문: `<작품명>.txt`
- 그래프: `<작품명>-KG.json`
- 일반 평가셋: `<작품명>-eval.jsonl`
- 하드 평가셋: `<작품명>-hardset.jsonl`
- 일반 평가 결과: `<작품명>-eval-results.json`
- 하드 평가 결과: `<작품명>-hardset-results.json`

이 규칙을 맞추면 여러 작품을 동일 스크립트/패턴으로 반복 실행하기 쉽습니다.
