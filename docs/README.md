# Tiny-Graph-RAG Documentation

이 문서는 Tiny-Graph-RAG 프로젝트의 내부 구조, 설계 원칙 및 핵심 알고리즘을 설명합니다.

## 1. 프로젝트 설계 원칙

Tiny-Graph-RAG는 복잡한 Graph DB 인프라 없이도 강력한 추론 능력을 가진 RAG 시스템을 구축하는 것을 목표로 합니다.

- **투명성 (Transparency)**: 벡터 임베딩의 블랙박스 검색 대신, 명시적인 그래프 탐색(BFS)을 통해 어떤 정보가 왜 선택되었는지 추적 가능하게 합니다.
- **경량화 (Lightweight)**: 별도의 서버 설치 없이 Python 라이브러리와 JSON 파일만으로 구동됩니다.
- **정밀도 (Precision)**: LLM을 활용한 Entity Resolution 및 랭킹 알고리즘을 통해 단순 키워드 매칭보다 높은 정밀도를 지향합니다.

## 2. 모듈별 아키텍처 및 책임

시스템은 `tiny_graph_rag/` 패키지 아래에 논리적으로 분리되어 있습니다.

| 모듈 | 책임 | 주요 기능 |
| :--- | :--- | :--- |
| **`chunking`** | 원본 문서 분할 | 중첩(Overlap)을 허용하는 문장/문단 기반 청킹 (`TextChunker`) |
| **`extraction`** | 지식 추출 | LLM을 이용해 청크에서 엔티티와 관계를 JSON 형태로 추출 |
| **`graph`** | 그래프 관리 | 엔티티 병합(Entity Resolution), 그래프 구축(Build), JSON 저장/로드 |
| **`retrieval`** | 정보 검색 | 질문 핵심 엔티티 추출, BFS 기반 주변 정보 탐색, 휴리스틱 랭킹 |
| **`llm`** | 언어 모델 연동 | OpenAI API 호출 최적화, 프롬프트 템플릿 관리, 비용 계산 |
| **`evaluation`** | 품질 평가 | 정답셋 기반의 검색 성능 측정 (Precision, Recall, MRR, nDCG) |
| **`visualization`** | 시각화 | 구축된 그래프를 대화형 HTML(Pyvis)로 렌더링 |

## 3. 핵심 알고리즘 상세

### 3.1 Retrieval 파이프라인
`GraphRetriever.retrieve()`는 다음과 같은 순서로 최적의 컨텍스트를 구성합니다.

1. **Entity Identification**: 질문에서 주요 엔티티(Seed Entities)를 추출합니다.
2. **Anchor Matching**: 그래프 내에서 Seed Entities와 일치하는 노드를 찾습니다. (Exact match 및 Fuzzy match 지원)
3. **Neighborhood Expansion**: 매칭된 노드로부터 설정된 `hops` 만큼 BFS 탐색을 수행하여 관련 있는 서브그래프를 확장합니다.
4. **Scoring & Ranking**: 탐색된 엔티티와 관계들에 대해 질문과의 관련성 점수를 계산하고 Top-K를 선정합니다.
5. **Context Assembly**: 선정된 지식 조각들을 LLM이 이해하기 쉬운 텍스트 형식으로 변환합니다.

### 3.2 Entity Resolution (ER)
동일한 대상을 가리키는 서로 다른 이름(예: '나', '인력거꾼', '김첨지')을 하나로 병합하여 그래프의 밀도를 높입니다.
- 상세 내용은 [Entity Resolution 가이드](entity-resolution.md)를 참고하세요.

## 4. 제약 사항 및 향후 과제

- **속도**: 그래프 구축 과정에서 많은 LLM 호출이 발생할 수 있습니다.
- **대규모 데이터**: 현재는 메모리 기반으로 동작하므로 수백만 노드 수준의 대규모 데이터 처리에는 적합하지 않습니다.
- **의미론적 유사도**: 현재 랭킹 시스템은 텍스트 포함 여부 기반의 휴리스틱을 사용하며, 벡터 임베딩 기반의 유사도 측정은 포함되어 있지 않습니다.

## 5. 추가 문서 안내

- [평가 가이드](evaluation.md): 데이터셋 형식 및 성능 평가 방법
- [Entity Resolution 가이드](entity-resolution.md): 엔티티 병합 알고리즘 상세
- [Chunking 가이드](chunking.md): 텍스트 분할 전략 및 설정

## 6. 저장 경로 기본값

`config.yaml`의 `storage` 섹션 기본값:
- `kg_dir`: `data/kg`
- `dataset_dir`: `data/eval`
- `results_dir`: `data/results`

CLI의 상대 경로는 위 기본 폴더를 기준으로 해석됩니다.
