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
        eid = graph.add_entity(entity)

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

        assert calf_id not in graph.entities
        assert kim_id in graph.entities
        assert "송아지" in graph.entities[kim_id].aliases

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
        assert found.name == "김첨지"

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
