# Tiny-Graph-RAG

Tiny-Graph-RAG는 OpenAI API를 이용해 텍스트에서 엔티티/관계를 추출하고, JSON 기반 지식 그래프를 만든 뒤 그래프 탐색으로 답변 컨텍스트를 구성하는 실험용 Graph RAG 프로젝트입니다.

벡터 DB 기반 검색 대신, 엔티티 연결 구조(BFS + 휴리스틱 랭킹)를 활용해 retrieval 과정을 투명하게 확인하는 데 초점을 둡니다.

## 프로젝트 범위

- 교육/실험 목적의 naive Graph RAG 구현
- 텍스트 문서 -> 지식 그래프(JSON) -> 질의/평가 파이프라인 제공
- OpenAI 호환 API(`OPENAI_BASE_URL`) 지원
- 노벨 데이터셋(`data/novels`) 기반 일반/하드셋 평가 지원

## 아키텍처 요약

```text
Document
  -> TextChunker
  -> EntityRelationshipExtractor (LLM JSON)
  -> ExtractionParser
  -> GraphBuilder / KnowledgeGraph
  -> GraphRetriever (query entity extraction -> BFS traversal -> ranking)
  -> LLM answer generation
```

핵심 모듈은 `tiny_graph_rag/` 아래에 있으며 상세 설명은 `docs/README.md`를 참고하세요.

## 빠른 시작

요구 사항: Python 3.13+, OpenAI API Key

```bash
uv sync
export OPENAI_API_KEY="your-api-key"
```

`config.yaml`로 기본 모델/청킹 설정을 관리하고, 환경변수가 최종 우선순위를 가집니다.

## 실행 방법 (CLI)

### 1) 문서에서 그래프 생성

```bash
uv run python main.py process "data/novels/김유정-동백꽃.txt" -o "data/novels/김유정-동백꽃-KG.json"
```

### 2) 그래프 질의

```bash
uv run python main.py query "점순이와 우리 수탉의 관계를 설명해줘." -g "data/novels/김유정-동백꽃-KG.json"
```

### 3) 그래프 통계 확인

```bash
uv run python main.py stats -g "data/novels/김유정-동백꽃-KG.json"
```

### 4) 시각화 HTML 생성

```bash
uv run python main.py visualize -g "data/novels/김유정-동백꽃-KG.json" -o graph_viz.html
```

### 5) Streamlit UI

```bash
uv run streamlit run streamlit_app.py
```

## 평가 워크플로우

평가는 `main.py eval`로 수행하며, 출력 JSON에는 예제별 메트릭과 전체 요약(지연 시간/토큰/예상 비용)이 저장됩니다.

### 기본(일반) 평가

```bash
uv run python main.py eval \
  --dataset "data/novels/김유정-동백꽃-eval.jsonl" \
  -g "data/novels/김유정-동백꽃-KG.json" \
  -o "data/novels/김유정-동백꽃-eval-results.json"
```

### Hardset 평가 (alias/multi-hop)

```bash
uv run python main.py eval \
  --dataset "data/novels/김유정-동백꽃-hardset.jsonl" \
  -g "data/novels/김유정-동백꽃-KG.json" \
  --hops 4 \
  -o "data/novels/김유정-동백꽃-hardset-results.json"
```

옵션:
- `--top-k`: top-k 기준 메트릭 계산 (기본 5)
- `--hops`: BFS 깊이 (기본 2)
- `--skip-generation`: 답변 생성 호출 생략(검색 품질만 측정)
- `--price-per-1k-input`, `--price-per-1k-output`: 비용 추정 단가

## 테스트

```bash
uv run pytest
```

## 데이터셋/결과 파일 규칙

`data/novels` 하위 파일은 아래 패턴을 따릅니다.

- 원문: `<작품명>.txt`
- 그래프: `<작품명>-KG.json`
- 일반 평가셋: `<작품명>-eval.jsonl`
- 하드 평가셋: `<작품명>-hardset.jsonl`
- 일반 평가 결과: `<작품명>-eval-results.json`
- 하드 평가 결과: `<작품명>-hardset-results.json`

예: `김유정-동백꽃-eval.jsonl`, `이상-날개-hardset-results.json`

## 문서

- 프로젝트/모듈 개요: `docs/README.md`
- 평가 데이터셋/실행 가이드: `docs/evaluation.md`

## 라이선스

MIT
