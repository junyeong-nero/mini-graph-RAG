"""Tests for knowledge graph models and operations."""

from unittest.mock import MagicMock

from tiny_graph_rag.graph import Entity, KnowledgeGraph, LLMEntityResolver, Relationship


class TestEntity:
    """Tests for Entity class."""

    def test_entity_creation(self):
        """Test creating an Entity."""
        entity = Entity(
            name="John Doe",
            entity_type="PERSON",
            description="A software engineer",
        )

        assert entity.name == "John Doe"
        assert entity.entity_type == "PERSON"
        assert entity.description == "A software engineer"
        assert entity.entity_id  # Should have auto-generated UUID

    def test_entity_merge(self):
        """Test merging two entities."""
        entity1 = Entity(
            name="John",
            entity_type="PERSON",
            description="Works at Acme",
            source_chunks=["chunk1"],
        )
        entity2 = Entity(
            name="John",
            entity_type="PERSON",
            description="Lives in NYC",
            source_chunks=["chunk2"],
        )

        merged = entity1.merge_with(entity2)

        assert merged.name == "John"
        assert "Works at Acme" in merged.description
        assert "Lives in NYC" in merged.description
        assert "chunk1" in merged.source_chunks
        assert "chunk2" in merged.source_chunks

    def test_entity_to_dict(self):
        """Test entity serialization."""
        entity = Entity(
            name="Test",
            entity_type="CONCEPT",
            description="A test entity",
        )

        data = entity.to_dict()

        assert data["name"] == "Test"
        assert data["entity_type"] == "CONCEPT"
        assert data["description"] == "A test entity"

    def test_entity_from_dict(self):
        """Test entity deserialization."""
        data = {
            "entity_id": "123",
            "name": "Test",
            "entity_type": "CONCEPT",
            "description": "A test",
        }

        entity = Entity.from_dict(data)

        assert entity.entity_id == "123"
        assert entity.name == "Test"
        assert entity.entity_type == "CONCEPT"


class TestRelationship:
    """Tests for Relationship class."""

    def test_relationship_creation(self):
        """Test creating a Relationship."""
        rel = Relationship(
            source_entity_id="entity1",
            target_entity_id="entity2",
            relationship_type="WORKS_FOR",
            description="John works for Acme",
        )

        assert rel.source_entity_id == "entity1"
        assert rel.target_entity_id == "entity2"
        assert rel.relationship_type == "WORKS_FOR"
        assert rel.relationship_id  # Should have auto-generated UUID

    def test_relationship_to_dict(self):
        """Test relationship serialization."""
        rel = Relationship(
            source_entity_id="e1",
            target_entity_id="e2",
            relationship_type="KNOWS",
        )

        data = rel.to_dict()

        assert data["source_entity_id"] == "e1"
        assert data["target_entity_id"] == "e2"
        assert data["relationship_type"] == "KNOWS"


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph class."""

    def test_empty_graph(self):
        """Test empty graph creation."""
        graph = KnowledgeGraph()

        assert len(graph.entities) == 0
        assert len(graph.relationships) == 0

    def test_add_entity(self):
        """Test adding an entity."""
        graph = KnowledgeGraph()
        entity = Entity(name="John", entity_type="PERSON")

        entity_id = graph.add_entity(entity)

        assert entity_id == entity.entity_id
        assert len(graph.entities) == 1
        assert graph.entities[entity_id].name == "John"

    def test_add_duplicate_entity(self):
        """Test that duplicate entities are merged."""
        graph = KnowledgeGraph()
        entity1 = Entity(name="John", entity_type="PERSON", description="First")
        entity2 = Entity(name="John", entity_type="PERSON", description="Second")

        id1 = graph.add_entity(entity1)
        id2 = graph.add_entity(entity2)

        assert id1 == id2
        assert len(graph.entities) == 1
        # Description should be merged
        merged = graph.entities[id1]
        assert "First" in merged.description
        assert "Second" in merged.description

    def test_get_entity_by_name(self):
        """Test looking up entity by name."""
        graph = KnowledgeGraph()
        entity = Entity(name="John Doe", entity_type="PERSON")
        graph.add_entity(entity)

        # Exact match
        found = graph.get_entity_by_name("John Doe")
        assert found is not None
        assert found.name == "John Doe"

        # Case insensitive
        found = graph.get_entity_by_name("john doe")
        assert found is not None

        # Not found
        found = graph.get_entity_by_name("Jane")
        assert found is None

    def test_add_relationship(self):
        """Test adding a relationship."""
        graph = KnowledgeGraph()
        rel = Relationship(
            source_entity_id="e1",
            target_entity_id="e2",
            relationship_type="KNOWS",
        )

        graph.add_relationship(rel)

        assert len(graph.relationships) == 1
        assert graph.relationships[0].relationship_type == "KNOWS"

    def test_get_neighbors(self):
        """Test getting neighboring entities."""
        graph = KnowledgeGraph()

        # Add entities
        e1 = Entity(name="A", entity_type="CONCEPT")
        e2 = Entity(name="B", entity_type="CONCEPT")
        e3 = Entity(name="C", entity_type="CONCEPT")

        id1 = graph.add_entity(e1)
        id2 = graph.add_entity(e2)
        id3 = graph.add_entity(e3)

        # Add relationships: A -> B -> C
        graph.add_relationship(Relationship(
            source_entity_id=id1,
            target_entity_id=id2,
            relationship_type="RELATED",
        ))
        graph.add_relationship(Relationship(
            source_entity_id=id2,
            target_entity_id=id3,
            relationship_type="RELATED",
        ))

        # 1 hop from A should get B
        neighbors_1 = graph.get_neighbors(id1, hops=1)
        assert id2 in neighbors_1

        # 2 hops from A should get B and C
        neighbors_2 = graph.get_neighbors(id1, hops=2)
        assert id2 in neighbors_2
        assert id3 in neighbors_2

    def test_get_relationships_for_entity(self):
        """Test getting relationships for an entity."""
        graph = KnowledgeGraph()

        e1 = Entity(name="A", entity_type="CONCEPT")
        e2 = Entity(name="B", entity_type="CONCEPT")

        id1 = graph.add_entity(e1)
        id2 = graph.add_entity(e2)

        rel = Relationship(
            source_entity_id=id1,
            target_entity_id=id2,
            relationship_type="RELATED",
        )
        graph.add_relationship(rel)

        # A's relationships
        rels_a = graph.get_relationships_for_entity(id1)
        assert len(rels_a) == 1

        # B's relationships (as target)
        rels_b = graph.get_relationships_for_entity(id2)
        assert len(rels_b) == 1

    def test_graph_serialization(self):
        """Test graph to_dict and from_dict."""
        graph = KnowledgeGraph()

        e1 = Entity(name="A", entity_type="CONCEPT")
        e2 = Entity(name="B", entity_type="CONCEPT")
        id1 = graph.add_entity(e1)
        id2 = graph.add_entity(e2)

        graph.add_relationship(Relationship(
            source_entity_id=id1,
            target_entity_id=id2,
            relationship_type="RELATED",
        ))

        # Serialize
        data = graph.to_dict()

        # Deserialize
        restored = KnowledgeGraph.from_dict(data)

        assert len(restored.entities) == 2
        assert len(restored.relationships) == 1
        assert restored.get_entity_by_name("A") is not None
        assert restored.get_entity_by_name("B") is not None

    def test_merge_entities_remaps_relationships(self):
        """Test merging entities remaps and deduplicates relationships."""
        graph = KnowledgeGraph()

        kim = Entity(name="김첨지", entity_type="PERSON")
        husband = Entity(name="남편", entity_type="PERSON")
        wife = Entity(name="아내", entity_type="PERSON")

        kim_id = graph.add_entity(kim)
        husband_id = graph.add_entity(husband)
        wife_id = graph.add_entity(wife)

        graph.add_relationship(
            Relationship(
                source_entity_id=kim_id,
                target_entity_id=wife_id,
                relationship_type="CARES_FOR",
            )
        )
        graph.add_relationship(
            Relationship(
                source_entity_id=husband_id,
                target_entity_id=wife_id,
                relationship_type="CARES_FOR",
            )
        )

        merged = graph.merge_entities(kim_id, husband_id)

        assert merged is True
        assert husband_id not in graph.entities
        assert len(graph.relationships) == 1
        assert graph.relationships[0].source_entity_id == kim_id
        assert graph.relationships[0].target_entity_id == wife_id


    def test_merge_entities_preserves_aliases(self):
        """Test that merge_entities records the duplicate's name as an alias."""
        graph = KnowledgeGraph()

        kim = Entity(name="김첨지", entity_type="PERSON")
        calf = Entity(name="송아지", entity_type="PERSON")

        kim_id = graph.add_entity(kim)
        calf_id = graph.add_entity(calf)

        graph.merge_entities(kim_id, calf_id)

        canonical = graph.entities[kim_id]
        assert "송아지" in canonical.aliases

    def test_lookup_by_alias_after_merge(self):
        """Test that entity can be found by alias name after merge."""
        graph = KnowledgeGraph()

        kim = Entity(name="김첨지", entity_type="PERSON")
        husband = Entity(name="남편", entity_type="PERSON")

        kim_id = graph.add_entity(kim)
        husband_id = graph.add_entity(husband)

        graph.merge_entities(kim_id, husband_id)

        found = graph.get_entity_by_name("남편")
        assert found is not None
        assert found.entity_id == kim_id

    def test_add_entity_with_aliases_deduplicates(self):
        """Test that adding entity whose alias matches existing entity merges them."""
        graph = KnowledgeGraph()

        kim = Entity(name="김첨지", entity_type="PERSON", description="인력거꾼")
        kim_id = graph.add_entity(kim)

        alias_entity = Entity(
            name="인력거꾼",
            entity_type="PERSON",
            description="같은 사람",
            aliases=["김첨지"],
        )
        alias_id = graph.add_entity(alias_entity)

        assert alias_id == kim_id
        assert len(graph.entities) == 1

    def test_entity_aliases_serialization_roundtrip(self):
        """Test that aliases survive to_dict -> from_dict roundtrip."""
        entity = Entity(
            name="김첨지",
            entity_type="PERSON",
            aliases=["송아지", "남편"],
        )

        data = entity.to_dict()
        assert data["aliases"] == ["송아지", "남편"]

        restored = Entity.from_dict(data)
        assert restored.aliases == ["송아지", "남편"]

    def test_graph_from_dict_indexes_aliases(self):
        """Test that KnowledgeGraph.from_dict indexes aliases for lookup."""
        graph = KnowledgeGraph()
        entity = Entity(name="김첨지", entity_type="PERSON", aliases=["송아지"])
        graph.add_entity(entity)

        data = graph.to_dict()
        restored = KnowledgeGraph.from_dict(data)

        found = restored.get_entity_by_name("송아지")
        assert found is not None
        assert found.name == "김첨지"


class TestEntityResolution:
    """Tests for LLM-based entity resolution."""

    def test_llm_entity_resolver_merges_aliases(self):
        """Test resolver can merge non-lexical aliases."""
        graph = KnowledgeGraph()

        kim = Entity(name="김첨지", entity_type="PERSON", description="인력거꾼")
        calf = Entity(name="송아지", entity_type="PERSON", description="같은 인물의 별명")
        wife = Entity(name="아내", entity_type="PERSON")

        kim_id = graph.add_entity(kim)
        calf_id = graph.add_entity(calf)
        wife_id = graph.add_entity(wife)

        graph.add_relationship(
            Relationship(
                source_entity_id=kim_id,
                target_entity_id=wife_id,
                relationship_type="HUSBAND_OF",
            )
        )
        graph.add_relationship(
            Relationship(
                source_entity_id=calf_id,
                target_entity_id=wife_id,
                relationship_type="HUSBAND_OF",
            )
        )

        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {
            "merge_groups": [
                {
                    "canonical_entity_id": kim_id,
                    "duplicate_entity_ids": [calf_id],
                    "confidence": 0.95,
                    "reason": "same person via nickname",
                }
            ]
        }

        resolver = LLMEntityResolver(llm_client=mock_llm)
        resolver.resolve(graph)

        assert len(graph.entities) == 2
        kim_found = graph.get_entity_by_name("김첨지")
        calf_found = graph.get_entity_by_name("송아지")
        assert kim_found is not None
        assert calf_found is not None
        assert kim_found.entity_id == calf_found.entity_id

    def test_resolver_preserves_aliases_and_lookup(self):
        """After LLM resolution, merged alias name is queryable."""
        graph = KnowledgeGraph()

        kim = Entity(name="김첨지", entity_type="PERSON", description="인력거꾼")
        calf = Entity(name="송아지", entity_type="PERSON", description="별명")

        kim_id = graph.add_entity(kim)
        calf_id = graph.add_entity(calf)

        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {
            "merge_groups": [
                {
                    "canonical_entity_id": kim_id,
                    "duplicate_entity_ids": [calf_id],
                    "confidence": 0.90,
                    "reason": "same person",
                }
            ]
        }

        resolver = LLMEntityResolver(llm_client=mock_llm)
        resolver.resolve(graph)

        found = graph.get_entity_by_name("송아지")
        assert found is not None
        kim_found = graph.get_entity_by_name("김첨지")
        assert kim_found is not None
        assert found.entity_id == kim_found.entity_id

    def test_resolver_skips_low_confidence(self):
        """Merge groups below min_confidence are ignored."""
        graph = KnowledgeGraph()

        e1 = Entity(name="A", entity_type="PERSON")
        e2 = Entity(name="B", entity_type="PERSON")
        e1_id = graph.add_entity(e1)
        e2_id = graph.add_entity(e2)

        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {
            "merge_groups": [
                {
                    "canonical_entity_id": e1_id,
                    "duplicate_entity_ids": [e2_id],
                    "confidence": 0.5,
                    "reason": "maybe same",
                }
            ]
        }

        resolver = LLMEntityResolver(llm_client=mock_llm, min_confidence=0.75)
        resolver.resolve(graph)

        assert e1_id in graph.entities
        assert e2_id in graph.entities

    def test_resolver_handles_llm_error(self):
        """Resolver gracefully handles LLM API errors."""
        graph = KnowledgeGraph()

        e1 = Entity(name="A", entity_type="PERSON")
        e2 = Entity(name="B", entity_type="PERSON")
        graph.add_entity(e1)
        graph.add_entity(e2)

        mock_llm = MagicMock()
        mock_llm.chat_json.side_effect = RuntimeError("API down")

        resolver = LLMEntityResolver(llm_client=mock_llm)
        resolver.resolve(graph)

        assert len(graph.entities) == 2

    def test_resolver_merges_transitive_groups(self):
        """Resolver should merge transitive groups even with intermediate canonical IDs."""
        graph = KnowledgeGraph()

        e1 = Entity(name="김첨지", entity_type="PERSON")
        e2 = Entity(name="남편", entity_type="PERSON")
        e3 = Entity(name="차부", entity_type="PERSON")
        e1_id = graph.add_entity(e1)
        e2_id = graph.add_entity(e2)
        e3_id = graph.add_entity(e3)

        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {
            "merge_groups": [
                {
                    "canonical_entity_id": e1_id,
                    "duplicate_entity_ids": [e2_id],
                    "confidence": 0.95,
                    "reason": "same person",
                },
                {
                    "canonical_entity_id": e2_id,
                    "duplicate_entity_ids": [e3_id],
                    "confidence": 0.95,
                    "reason": "same person",
                },
            ]
        }

        resolver = LLMEntityResolver(llm_client=mock_llm)
        resolver.resolve(graph)

        assert len(graph.entities) == 1
        found = graph.get_entity_by_name("차부")
        assert found is not None
        assert found.entity_id == e1_id

    def test_resolver_merges_person_like_other_entities(self):
        """Resolver should merge OTHER entities when they are person-like mentions."""
        graph = KnowledgeGraph()

        wife = Entity(name="아내", entity_type="PERSON")
        patient = Entity(name="병인", entity_type="OTHER", description="집의 병자")
        insult = Entity(name="오라질년", entity_type="PERSON")

        wife_id = graph.add_entity(wife)
        patient_id = graph.add_entity(patient)
        insult_id = graph.add_entity(insult)

        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {
            "merge_groups": [
                {
                    "canonical_entity_id": wife_id,
                    "duplicate_entity_ids": [patient_id, insult_id],
                    "confidence": 0.95,
                    "reason": "same patient wife",
                }
            ]
        }

        resolver = LLMEntityResolver(llm_client=mock_llm)
        resolver.resolve(graph)

        assert len(graph.entities) == 1
        wife_found = graph.get_entity_by_name("아내")
        patient_found = graph.get_entity_by_name("병인")
        insult_found = graph.get_entity_by_name("오라질년")
        assert wife_found is not None
        assert patient_found is not None
        assert insult_found is not None
        assert wife_found.entity_id == patient_found.entity_id
        assert wife_found.entity_id == insult_found.entity_id

    def test_resolver_contextual_merge_via_llm_signals(self):
        """LLM receives shared-neighbor signals and returns merge group for wife/patient."""
        graph = KnowledgeGraph()

        kim = Entity(name="김첨지", entity_type="PERSON")
        wife = Entity(name="아내", entity_type="PERSON")
        patient = Entity(name="병인", entity_type="PERSON", description="집의 환자")

        kim_id = graph.add_entity(kim)
        wife_id = graph.add_entity(wife)
        patient_id = graph.add_entity(patient)

        graph.add_relationship(
            Relationship(
                source_entity_id=kim_id,
                target_entity_id=wife_id,
                relationship_type="MARRIED_TO",
            )
        )
        graph.add_relationship(
            Relationship(
                source_entity_id=kim_id,
                target_entity_id=patient_id,
                relationship_type="CARES_FOR",
            )
        )
        graph.add_relationship(
            Relationship(
                source_entity_id=patient_id,
                target_entity_id=wife_id,
                relationship_type="RELATED_TO",
            )
        )

        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {
            "merge_groups": [
                {
                    "canonical_entity_id": wife_id,
                    "duplicate_entity_ids": [patient_id],
                    "confidence": 0.90,
                    "reason": "shared neighbor 김첨지 + patient role",
                }
            ]
        }

        resolver = LLMEntityResolver(llm_client=mock_llm)
        resolver.resolve(graph)

        wife_found = graph.get_entity_by_name("아내")
        patient_found = graph.get_entity_by_name("병인")
        assert wife_found is not None
        assert patient_found is not None
        assert wife_found.entity_id == patient_found.entity_id

        # Verify LLM prompt included candidate signals
        call_args = mock_llm.chat_json.call_args
        user_prompt = call_args.kwargs.get("user_prompt", call_args[1].get("user_prompt", "")) if call_args.kwargs else ""
        if not user_prompt and call_args.args:
            user_prompt = call_args.kwargs.get("user_prompt", "")
        assert "Candidate merge pairs" in user_prompt

    def test_resolver_role_bucket_merge_via_llm_signals(self):
        """LLM receives shared-neighbor signals for spouse-bucket pair and returns merge group."""
        graph = KnowledgeGraph()

        kim = Entity(name="김첨지", entity_type="PERSON")
        wife = Entity(name="아내", entity_type="PERSON")
        patient = Entity(name="병인", entity_type="PERSON", description="김첨지의 아내")

        kim_id = graph.add_entity(kim)
        wife_id = graph.add_entity(wife)
        patient_id = graph.add_entity(patient)

        graph.add_relationship(
            Relationship(
                source_entity_id=kim_id,
                target_entity_id=wife_id,
                relationship_type="MARRIED_TO",
            )
        )
        graph.add_relationship(
            Relationship(
                source_entity_id=kim_id,
                target_entity_id=patient_id,
                relationship_type="CARES_FOR",
            )
        )

        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {
            "merge_groups": [
                {
                    "canonical_entity_id": wife_id,
                    "duplicate_entity_ids": [patient_id],
                    "confidence": 0.90,
                    "reason": "same role bucket spouse + shared neighbor",
                }
            ]
        }

        resolver = LLMEntityResolver(llm_client=mock_llm)
        resolver.resolve(graph)

        wife_found = graph.get_entity_by_name("아내")
        patient_found = graph.get_entity_by_name("병인")
        assert wife_found is not None
        assert patient_found is not None
        assert wife_found.entity_id == patient_found.entity_id

    def test_collect_merge_signals_includes_role_bucket(self):
        """_collect_merge_signals returns shared_neighbors signal for entities sharing a neighbor."""
        graph = KnowledgeGraph()

        kim = Entity(name="김첨지", entity_type="PERSON")
        wife = Entity(name="아내", entity_type="PERSON")
        patient = Entity(name="병인", entity_type="PERSON", description="김첨지의 아내")

        kim_id = graph.add_entity(kim)
        wife_id = graph.add_entity(wife)
        patient_id = graph.add_entity(patient)

        graph.add_relationship(
            Relationship(
                source_entity_id=kim_id,
                target_entity_id=wife_id,
                relationship_type="MARRIED_TO",
            )
        )
        graph.add_relationship(
            Relationship(
                source_entity_id=kim_id,
                target_entity_id=patient_id,
                relationship_type="CARES_FOR",
            )
        )

        mock_llm = MagicMock()
        resolver = LLMEntityResolver(llm_client=mock_llm)

        entities = [
            graph.entities[wife_id],
            graph.entities[patient_id],
        ]
        signals = resolver._collect_merge_signals(graph, entities)

        assert len(signals) >= 1
        pair = signals[0]
        assert "shared_neighbors" in pair
        neighbor_names = [n["neighbor_name"] for n in pair["shared_neighbors"]]
        assert "김첨지" in neighbor_names

    def test_resolver_does_not_merge_married_pair_even_if_llm_suggests(self):
        """Direct MARRIED_TO relation must block erroneous LLM merge."""
        graph = KnowledgeGraph()

        kim = Entity(name="김첨지", entity_type="PERSON")
        wife = Entity(name="아내", entity_type="PERSON")
        kim_id = graph.add_entity(kim)
        wife_id = graph.add_entity(wife)

        graph.add_relationship(
            Relationship(
                source_entity_id=kim_id,
                target_entity_id=wife_id,
                relationship_type="MARRIED_TO",
            )
        )

        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {
            "merge_groups": [
                {
                    "canonical_entity_id": kim_id,
                    "duplicate_entity_ids": [wife_id],
                    "confidence": 0.99,
                    "reason": "incorrect merge",
                }
            ]
        }

        resolver = LLMEntityResolver(llm_client=mock_llm)
        resolver.resolve(graph)

        kim_found = graph.get_entity_by_name("김첨지")
        wife_found = graph.get_entity_by_name("아내")
        assert kim_found is not None
        assert wife_found is not None
        assert kim_found.entity_id != wife_found.entity_id

    def test_collect_merge_signals_co_occurring_chunks(self):
        """co_occurring_chunks signal is present when entities share source chunks."""
        graph = KnowledgeGraph()

        e1 = Entity(
            name="A",
            entity_type="PERSON",
            source_chunks=["chunk1", "chunk2", "chunk3"],
        )
        e2 = Entity(
            name="B",
            entity_type="PERSON",
            source_chunks=["chunk2", "chunk3", "chunk4"],
        )

        graph.add_entity(e1)
        graph.add_entity(e2)

        mock_llm = MagicMock()
        resolver = LLMEntityResolver(llm_client=mock_llm)

        signals = resolver._collect_merge_signals(graph, [e1, e2])

        assert len(signals) == 1
        pair = signals[0]
        assert "co_occurring_chunks" in pair
        assert pair["co_occurring_chunks"] == ["chunk2", "chunk3"]

    def test_resolve_batch_includes_candidate_signals_in_prompt(self):
        """LLM prompt includes 'Candidate merge pairs' section when signals exist."""
        graph = KnowledgeGraph()

        kim = Entity(name="김첨지", entity_type="PERSON")
        wife = Entity(name="아내", entity_type="PERSON")
        patient = Entity(name="병인", entity_type="PERSON")

        kim_id = graph.add_entity(kim)
        wife_id = graph.add_entity(wife)
        patient_id = graph.add_entity(patient)

        graph.add_relationship(
            Relationship(
                source_entity_id=kim_id,
                target_entity_id=wife_id,
                relationship_type="MARRIED_TO",
            )
        )
        graph.add_relationship(
            Relationship(
                source_entity_id=kim_id,
                target_entity_id=patient_id,
                relationship_type="CARES_FOR",
            )
        )

        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {"merge_groups": []}

        resolver = LLMEntityResolver(llm_client=mock_llm)
        batch = [graph.entities[wife_id], graph.entities[patient_id]]
        resolver._resolve_batch(graph, batch)

        call_args = mock_llm.chat_json.call_args
        user_prompt = call_args.kwargs.get("user_prompt", "")
        assert "Candidate merge pairs with supporting evidence" in user_prompt
        assert "shared_neighbors" in user_prompt
