"""Tests for entity and relationship extraction."""

import pytest

from mini_graph_rag.extraction.parser import ExtractionParser
from mini_graph_rag.graph import Entity, Relationship


class TestExtractionParser:
    """Tests for ExtractionParser class."""

    def test_parse_valid_response(self):
        """Test parsing a valid LLM response."""
        parser = ExtractionParser()

        response = {
            "entities": [
                {"name": "John", "type": "PERSON", "description": "A developer"},
                {"name": "Acme Corp", "type": "ORGANIZATION", "description": "A tech company"},
            ],
            "relationships": [
                {
                    "source": "John",
                    "target": "Acme Corp",
                    "type": "WORKS_FOR",
                    "description": "John works at Acme",
                }
            ],
        }

        entities, relationships = parser.parse(response, chunk_id="test_chunk")

        assert len(entities) == 2
        assert len(relationships) == 1

        # Check entities
        john = next(e for e in entities if e.name == "John")
        assert john.entity_type == "PERSON"
        assert john.description == "A developer"
        assert "test_chunk" in john.source_chunks

        # Check relationships
        rel = relationships[0]
        assert rel.relationship_type == "WORKS_FOR"

    def test_parse_empty_response(self):
        """Test parsing an empty response."""
        parser = ExtractionParser()

        response = {}

        entities, relationships = parser.parse(response)

        assert entities == []
        assert relationships == []

    def test_parse_missing_entity_name(self):
        """Test that entities without names are skipped."""
        parser = ExtractionParser()

        response = {
            "entities": [
                {"name": "", "type": "PERSON"},
                {"name": "Valid", "type": "PERSON"},
            ],
        }

        entities, _ = parser.parse(response)

        assert len(entities) == 1
        assert entities[0].name == "Valid"

    def test_parse_invalid_entity_type(self):
        """Test that invalid entity types default to OTHER."""
        parser = ExtractionParser()

        response = {
            "entities": [
                {"name": "Test", "type": "INVALID_TYPE"},
            ],
        }

        entities, _ = parser.parse(response)

        assert len(entities) == 1
        assert entities[0].entity_type == "OTHER"

    def test_parse_missing_relationship_endpoints(self):
        """Test that relationships with missing endpoints are skipped."""
        parser = ExtractionParser()

        response = {
            "entities": [
                {"name": "John", "type": "PERSON"},
            ],
            "relationships": [
                {"source": "John", "target": "Unknown", "type": "KNOWS"},
                {"source": "", "target": "John", "type": "KNOWS"},
            ],
        }

        entities, relationships = parser.parse(response)

        assert len(entities) == 1
        assert len(relationships) == 0  # Both should be skipped

    def test_parse_relationship_type_normalization(self):
        """Test that relationship types are normalized."""
        parser = ExtractionParser()

        response = {
            "entities": [
                {"name": "A", "type": "CONCEPT"},
                {"name": "B", "type": "CONCEPT"},
            ],
            "relationships": [
                {"source": "A", "target": "B", "type": "works for"},
            ],
        }

        entities, relationships = parser.parse(response)

        assert len(relationships) == 1
        assert relationships[0].relationship_type == "WORKS_FOR"

    def test_parse_case_insensitive_entity_matching(self):
        """Test that relationship entity matching is case insensitive."""
        parser = ExtractionParser()

        response = {
            "entities": [
                {"name": "John Doe", "type": "PERSON"},
                {"name": "Acme Corp", "type": "ORGANIZATION"},
            ],
            "relationships": [
                {
                    "source": "john doe",  # lowercase
                    "target": "ACME CORP",  # uppercase
                    "type": "WORKS_FOR",
                }
            ],
        }

        entities, relationships = parser.parse(response)

        assert len(entities) == 2
        assert len(relationships) == 1
