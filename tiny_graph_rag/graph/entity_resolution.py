"""LLM-based entity resolution for cross-mention alias merges."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..llm import OpenAIClient
from .models import Entity, KnowledgeGraph


ENTITY_RESOLUTION_SYSTEM_PROMPT = """You resolve whether extracted person-like entities refer to the same real-world character.

You will receive a list of person-like entities from one story. Each entity may have:
- name: the canonical name
- entity_type: PERSON or OTHER (OTHER can still be a person mention like patient/husband)
- aliases: known alternative names already identified during extraction
- description: context about the entity
- neighbors: relationship signals (type, other entity name/type, and description)

Some entries are aliases, role names, or nicknames with NO lexical overlap with the canonical name.
Examples: full name vs title, role label (남편/husband, 인력거꾼/driver), metaphorical nickname (송아지/calf used for a person).

Use neighbor overlap as strong evidence: if two entities share the same relationship partners with
the same relationship types, they are likely the same person.

Return strict JSON with this shape:
{
  "merge_groups": [
    {
      "canonical_entity_id": "id_to_keep",
      "duplicate_entity_ids": ["id_to_merge", "..."],
      "confidence": 0.0,
      "reason": "short reason"
    }
  ]
}

Rules:
- Merge only entities that refer to humans/characters.
- Prefer the entity with the most specific proper name as canonical.
- Use descriptions, neighbor signals, and existing aliases as evidence.
- Two names can be the same person even with zero string similarity.
- Do not invent IDs; use only provided IDs.
- Do not include canonical ID inside duplicate list.
- If unsure, do not merge.
"""



@dataclass(frozen=True)
class RoleBucket:
    """동일 인물로 병합할 수 있는 역할 용어 그룹."""

    name: str
    terms: frozenset[str]
    reassign_aliases: bool = False


@dataclass(frozen=True)
class EntityResolutionConfig:
    """Entity resolution에 사용되는 도메인/언어별 설정."""

    person_like_keywords: frozenset[str]
    generic_role_terms: frozenset[str]
    role_buckets: tuple[RoleBucket, ...]
    non_merge_relation_types: frozenset[str]


def default_config() -> EntityResolutionConfig:
    """기본 한국어 소설용 설정."""
    spouse_patient_terms = frozenset({
        "아내",
        "마누라",
        "그의 아내",
        "병인",
        "병자",
        "환자",
        "앓는 이",
        "이 환자",
        "오라질년",
        "오라질 년",
    })
    return EntityResolutionConfig(
        person_like_keywords=frozenset({
            "아내",
            "마누라",
            "남편",
            "환자",
            "병자",
            "병인",
            "오라질",
            "주정꾼",
            "주정뱅이",
            "인력거꾼",
            "차부",
            "wife",
            "husband",
            "patient",
            "driver",
            "drunkard",
        }),
        generic_role_terms=frozenset({
            "아내",
            "마누라",
            "남편",
            "환자",
            "병자",
            "병인",
            "주정꾼",
            "주정뱅이",
            "인력거꾼",
            "차부",
            "그",
            "그녀",
            "이 사람",
        }),
        role_buckets=(
            RoleBucket("spouse_patient", spouse_patient_terms, reassign_aliases=True),
        ),
        non_merge_relation_types=frozenset({
            "MARRIED_TO",
            "PARENT_OF",
            "CHILD_OF",
            "SIBLING_OF",
            "FRIEND_OF",
            "KNOWS",
        }),
    )


@dataclass
class LLMEntityResolver:
    """Resolve duplicate entities using a global LLM pass."""

    llm_client: OpenAIClient
    min_confidence: float = 0.75
    max_entities_per_pass: int = 80
    config: EntityResolutionConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = default_config()
        pattern = "|".join(
            re.escape(t)
            for t in sorted(self.config.generic_role_terms, key=len, reverse=True)
        )
        self._generic_role_pattern = re.compile(rf"^({pattern})$")

    def resolve(self, graph: KnowledgeGraph) -> None:
        """Resolve duplicate person-like entities in-place."""
        self._merge_explicit_alias_relationships(graph)
        self._merge_strong_contextual_aliases(graph)
        self._merge_role_bucket_aliases(graph)
        self._reassign_conflicting_aliases(graph)

        person_like_entities = [
            entity
            for entity in graph.entities.values()
            if self._is_person_like_entity(entity)
        ]

        if len(person_like_entities) < 2:
            return

        # Resolve in chunks to avoid oversized prompts.
        for start in range(0, len(person_like_entities), self.max_entities_per_pass):
            batch = person_like_entities[start:start + self.max_entities_per_pass]
            merge_groups = self._resolve_batch(graph, batch)
            self._apply_merge_groups(graph, merge_groups)

    def _resolve_batch(
        self,
        graph: KnowledgeGraph,
        entities: list[Entity],
    ) -> list[dict]:
        payload = []
        for entity in entities:
            entry: dict = {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "description": entity.description,
                "source_chunks": entity.source_chunks,
                "neighbors": self._get_neighbor_signals(graph, entity.entity_id),
            }
            if entity.aliases:
                entry["aliases"] = entity.aliases
            payload.append(entry)

        user_prompt = (
            "Resolve duplicate person-like entities from the following JSON array. "
            "Two names can still be the same person even with no lexical overlap if context/relations match.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

        try:
            response = self.llm_client.chat_json(
                system_prompt=ENTITY_RESOLUTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception:
            return []

        merge_groups = response.get("merge_groups", [])
        if not isinstance(merge_groups, list):
            return []
        return [group for group in merge_groups if isinstance(group, dict)]

    def _get_neighbor_signals(
        self,
        graph: KnowledgeGraph,
        entity_id: str,
    ) -> list[dict]:
        signals: list[dict] = []
        for rel in graph.get_relationships_for_entity(entity_id):
            other_id = (
                rel.target_entity_id
                if rel.source_entity_id == entity_id
                else rel.source_entity_id
            )
            other = graph.get_entity(other_id)
            if not other:
                continue
            signal: dict = {
                "relation_type": rel.relationship_type,
                "other_name": other.name,
                "other_type": other.entity_type,
            }
            if rel.description:
                signal["relation_desc"] = rel.description
            signals.append(signal)

        return signals[:12]

    def _apply_merge_groups(self, graph: KnowledgeGraph, merge_groups: list[dict]) -> None:
        parent: dict[str, str] = {}
        canonical_votes: dict[str, int] = {}

        def find(node: str) -> str:
            root = parent.setdefault(node, node)
            if root != node:
                root = find(root)
                parent[node] = root
            return root

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for group in merge_groups:
            canonical_id = group.get("canonical_entity_id")
            duplicate_ids = group.get("duplicate_entity_ids", [])
            confidence = float(group.get("confidence", 0.0))

            if confidence < self.min_confidence:
                continue
            if not isinstance(canonical_id, str):
                continue
            if not isinstance(duplicate_ids, list):
                continue

            canonical = graph.get_entity(canonical_id)
            if not canonical or not self._is_person_like_entity(canonical):
                continue

            canonical_votes[canonical_id] = canonical_votes.get(canonical_id, 0) + 1

            for duplicate_id in duplicate_ids:
                if not isinstance(duplicate_id, str):
                    continue

                duplicate = graph.get_entity(duplicate_id)
                if not duplicate or not self._is_person_like_entity(duplicate):
                    continue
                if self._is_non_mergeable_pair(graph, canonical_id, duplicate_id):
                    continue

                parent.setdefault(canonical_id, canonical_id)
                parent.setdefault(duplicate_id, duplicate_id)
                union(canonical_id, duplicate_id)

        clusters: dict[str, list[str]] = {}
        for entity_id in parent:
            clusters.setdefault(find(entity_id), []).append(entity_id)

        for cluster_entity_ids in clusters.values():
            if len(cluster_entity_ids) < 2:
                continue

            canonical_id = self._select_canonical_id(
                graph,
                cluster_entity_ids,
                canonical_votes,
            )
            if not canonical_id:
                continue

            for entity_id in cluster_entity_ids:
                if entity_id == canonical_id:
                    continue
                if not graph.get_entity(entity_id):
                    continue
                graph.merge_entities(canonical_id, entity_id)

    def _merge_explicit_alias_relationships(self, graph: KnowledgeGraph) -> None:
        alias_edges: list[tuple[str, str]] = []
        for rel in graph.relationships:
            rel_type = rel.relationship_type.upper()
            if rel_type not in {"ALIAS_OF", "SAME_AS"}:
                continue

            source = graph.get_entity(rel.source_entity_id)
            target = graph.get_entity(rel.target_entity_id)
            if not source or not target:
                continue
            if not self._is_person_like_entity(source):
                continue
            if not self._is_person_like_entity(target):
                continue

            alias_edges.append((source.entity_id, target.entity_id))

        for left_id, right_id in alias_edges:
            left = graph.get_entity(left_id)
            right = graph.get_entity(right_id)
            if not left or not right:
                continue

            canonical_id = self._select_canonical_id(graph, [left_id, right_id])
            if not canonical_id:
                continue
            duplicate_id = right_id if canonical_id == left_id else left_id
            graph.merge_entities(canonical_id, duplicate_id)

    def _merge_strong_contextual_aliases(self, graph: KnowledgeGraph) -> None:
        """Merge person-like aliases using strong contextual signals.

        This catches cases like 아내 == 병인 when extraction produced vague RELATED_TO links
        and shared third-party context, but no explicit ALIAS_OF relation.
        """
        changed = True
        while changed:
            changed = False
            entity_ids = list(graph.entities.keys())
            for i, left_id in enumerate(entity_ids):
                left = graph.get_entity(left_id)
                if not left or not self._is_person_like_entity(left):
                    continue

                for right_id in entity_ids[i + 1:]:
                    right = graph.get_entity(right_id)
                    if not right or not self._is_person_like_entity(right):
                        continue

                    direct_types = self._get_direct_relation_types(graph, left_id, right_id)
                    if not direct_types.intersection({"RELATED_TO", "REFERS_TO", "SAME_AS", "ALIAS_OF"}):
                        continue

                    if self._count_shared_neighbors(graph, left_id, right_id) < 1:
                        continue

                    if not (self._looks_reference_like(left) or self._looks_reference_like(right)):
                        continue

                    canonical_id = self._select_canonical_id(graph, [left_id, right_id])
                    if not canonical_id:
                        continue

                    duplicate_id = right_id if canonical_id == left_id else left_id
                    if graph.merge_entities(canonical_id, duplicate_id):
                        changed = True
                        break

                if changed:
                    break

    def _is_person_like_entity(self, entity: Entity) -> bool:
        if entity.entity_type == "PERSON":
            return True

        text = f"{entity.name} {entity.description}".lower()
        return any(keyword in text for keyword in self.config.person_like_keywords)

    def _role_bucket(self, entity: Entity) -> str | None:
        text = f"{entity.name} {entity.description}".lower()
        for bucket in self.config.role_buckets:
            if any(term in text for term in bucket.terms):
                return bucket.name
        return None

    def _is_bucket_term(self, value: str, bucket: RoleBucket) -> bool:
        normalized = value.lower().strip()
        return any(term in normalized for term in bucket.terms)

    def _merge_role_bucket_aliases(self, graph: KnowledgeGraph) -> None:
        """Merge role labels in the same semantic bucket with shared context."""
        changed = True
        while changed:
            changed = False
            entity_ids = list(graph.entities.keys())
            for i, left_id in enumerate(entity_ids):
                left = graph.get_entity(left_id)
                if not left or not self._is_person_like_entity(left):
                    continue

                left_bucket = self._role_bucket(left)
                if not left_bucket:
                    continue

                for right_id in entity_ids[i + 1:]:
                    right = graph.get_entity(right_id)
                    if not right or not self._is_person_like_entity(right):
                        continue

                    right_bucket = self._role_bucket(right)
                    if left_bucket != right_bucket:
                        continue
                    if self._count_shared_neighbors(graph, left_id, right_id) < 1:
                        continue

                    canonical_id = self._select_canonical_id(graph, [left_id, right_id])
                    if not canonical_id:
                        continue

                    duplicate_id = right_id if canonical_id == left_id else left_id
                    if graph.merge_entities(canonical_id, duplicate_id):
                        changed = True
                        break

                if changed:
                    break

    def _reassign_conflicting_aliases(self, graph: KnowledgeGraph) -> None:
        reassign_buckets = [b for b in self.config.role_buckets if b.reassign_aliases]
        if not reassign_buckets:
            return

        changed = False
        for bucket in reassign_buckets:
            bucket_entity_ids = [
                entity_id
                for entity_id, entity in graph.entities.items()
                if self._role_bucket(entity) == bucket.name
            ]
            if not bucket_entity_ids:
                continue

            canonical_id = self._select_canonical_id(graph, bucket_entity_ids)
            if not canonical_id:
                continue

            canonical_entity = graph.get_entity(canonical_id)
            if not canonical_entity:
                continue

            for entity_id, entity in graph.entities.items():
                if entity_id == canonical_id:
                    continue
                if not entity.aliases:
                    continue
                if self._role_bucket(entity) == bucket.name:
                    continue

                moved_aliases = [
                    alias
                    for alias in entity.aliases
                    if self._is_bucket_term(alias, bucket)
                ]
                if not moved_aliases:
                    continue

                entity.aliases = [
                    alias for alias in entity.aliases if alias not in moved_aliases
                ]
                for alias in moved_aliases:
                    if alias == canonical_entity.name:
                        continue
                    if alias not in canonical_entity.aliases:
                        canonical_entity.aliases.append(alias)
                changed = True

        if changed:
            graph._rebuild_entity_name_index()

    def _looks_reference_like(self, entity: Entity) -> bool:
        if entity.entity_type == "OTHER":
            return True
        if self._generic_role_pattern.match(entity.name):
            return True
        text = f"{entity.name} {entity.description}".lower()
        return any(keyword in text for keyword in self.config.person_like_keywords)

    def _get_direct_relation_types(
        self,
        graph: KnowledgeGraph,
        left_id: str,
        right_id: str,
    ) -> set[str]:
        relation_types: set[str] = set()
        for rel in graph.relationships:
            is_direct = (
                rel.source_entity_id == left_id and rel.target_entity_id == right_id
            ) or (
                rel.source_entity_id == right_id and rel.target_entity_id == left_id
            )
            if is_direct:
                relation_types.add(rel.relationship_type.upper())
        return relation_types

    def _count_shared_neighbors(
        self,
        graph: KnowledgeGraph,
        left_id: str,
        right_id: str,
    ) -> int:
        left_neighbors = graph.get_neighbors(left_id, hops=1) - {right_id}
        right_neighbors = graph.get_neighbors(right_id, hops=1) - {left_id}
        return len(left_neighbors.intersection(right_neighbors))

    def _is_non_mergeable_pair(
        self,
        graph: KnowledgeGraph,
        left_id: str,
        right_id: str,
    ) -> bool:
        direct_types = self._get_direct_relation_types(graph, left_id, right_id)
        return bool(direct_types.intersection(self.config.non_merge_relation_types))

    def _select_canonical_id(
        self,
        graph: KnowledgeGraph,
        entity_ids: list[str],
        canonical_votes: dict[str, int] | None = None,
    ) -> str | None:
        candidates: list[tuple[tuple[int, int, int, int], str]] = []
        votes = canonical_votes or {}
        for entity_id in entity_ids:
            entity = graph.get_entity(entity_id)
            if not entity:
                continue

            vote_score = votes.get(entity_id, 0)
            person_bonus = 1 if entity.entity_type == "PERSON" else 0
            role_penalty = 0 if not self._generic_role_pattern.match(entity.name) else -1
            relation_count = len(graph.get_relationships_for_entity(entity_id))
            score = (vote_score, person_bonus, role_penalty, relation_count)
            candidates.append((score, entity_id))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        return candidates[0][1]
