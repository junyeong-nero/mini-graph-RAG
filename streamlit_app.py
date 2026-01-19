"""Streamlit app for Mini-Graph-RAG visualization and interaction."""

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

from mini_graph_rag import GraphRAG
from mini_graph_rag.graph.storage import GraphStorage
from mini_graph_rag.graph.models import KnowledgeGraph


ENTITY_COLORS = {
    "PERSON": "#3498db",  # Blue
    "ORGANIZATION": "#2ecc71",  # Green
    "PLACE": "#e67e22",  # Orange
    "CONCEPT": "#9b59b6",  # Purple
    "EVENT": "#e74c3c",  # Red
    "OTHER": "#95a5a6",  # Gray
}

SELECTED_COLOR = "#FFD700"  # Gold


def load_graph(graph_path: str) -> KnowledgeGraph:
    storage = GraphStorage()
    return storage.load_json(graph_path)


def create_agraph_data(
    graph: KnowledgeGraph,
    filter_types: list[str] | None = None,
    max_nodes: int = 200,
    selected_entity_id: str | None = None,
) -> tuple[list[Node], list[Edge]]:
    nodes = []
    edges = []

    filtered_entities = {
        entity_id: entity
        for entity_id, entity in graph.entities.items()
        if not filter_types or entity.entity_type in filter_types
    }

    degrees = {entity_id: 0 for entity_id in filtered_entities}
    for rel in graph.relationships:
        if (
            rel.source_entity_id in filtered_entities
            and rel.target_entity_id in filtered_entities
        ):
            degrees[rel.source_entity_id] = degrees.get(rel.source_entity_id, 0) + 1
            degrees[rel.target_entity_id] = degrees.get(rel.target_entity_id, 0) + 1

    if len(filtered_entities) > max_nodes:
        sorted_ids = sorted(
            filtered_entities.keys(), key=lambda x: degrees.get(x, 0), reverse=True
        )
        limited_ids = set(sorted_ids[:max_nodes])
        filtered_entities = {
            k: v for k, v in filtered_entities.items() if k in limited_ids
        }

    for entity_id, entity in filtered_entities.items():
        color = ENTITY_COLORS.get(entity.entity_type, ENTITY_COLORS["OTHER"])
        size = min(15 + degrees.get(entity_id, 0) * 3, 50)

        if selected_entity_id and entity_id == selected_entity_id:
            color = SELECTED_COLOR
            size = size * 1.5

        title = f"{entity.name}\n[{entity.entity_type}]"
        if entity.description:
            title += f"\n{entity.description[:100]}..."

        nodes.append(
            Node(
                id=entity_id,
                label=entity.name,
                size=size,
                color=color,
                title=title,
            )
        )

    entity_ids = set(filtered_entities.keys())
    for rel in graph.relationships:
        if rel.source_entity_id in entity_ids and rel.target_entity_id in entity_ids:
            edges.append(
                Edge(
                    source=rel.source_entity_id,
                    target=rel.target_entity_id,
                    label=rel.relationship_type,
                    color="#888888",
                )
            )

    return nodes, edges


def get_entity_details(graph: KnowledgeGraph, entity_id: str) -> dict:
    entity = graph.get_entity(entity_id)
    if not entity:
        return {}

    relationships = graph.get_relationships_for_entity(entity_id)

    outgoing = []
    incoming = []
    for rel in relationships:
        if rel.source_entity_id == entity_id:
            target = graph.get_entity(rel.target_entity_id)
            if target:
                outgoing.append(
                    {
                        "type": rel.relationship_type,
                        "target": target.name,
                        "description": rel.description,
                    }
                )
        else:
            source = graph.get_entity(rel.source_entity_id)
            if source:
                incoming.append(
                    {
                        "type": rel.relationship_type,
                        "source": source.name,
                        "description": rel.description,
                    }
                )

    return {
        "entity": entity,
        "outgoing": outgoing,
        "incoming": incoming,
    }


def render_sidebar():
    with st.sidebar:
        st.header("📁 Load Graph")

        graph_path = st.text_input(
            "Graph JSON Path",
            value="data/novels/현진건-운수좋은날-KG.json",
            help="Path to the knowledge graph JSON file",
        )

        load_button = st.button("Load Graph", type="primary")

        st.divider()

        st.header("🎛️ Visualization Options")

        all_types = list(ENTITY_COLORS.keys())
        selected_types = st.multiselect(
            "Filter Entity Types",
            options=all_types,
            default=all_types,
            help="Select which entity types to display",
        )

        max_nodes = st.slider(
            "Max Nodes",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Maximum number of nodes to display",
        )

        st.divider()

        st.header("📊 Graph Stats")
        stats_placeholder = st.empty()

    return graph_path, load_button, selected_types, max_nodes, stats_placeholder


def render_stats(stats_placeholder, graph: KnowledgeGraph):
    with stats_placeholder:
        st.metric("Entities", len(graph.entities))
        st.metric("Relationships", len(graph.relationships))

        type_counts = {}
        for entity in graph.entities.values():
            type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1

        st.markdown("**Entity Types:**")
        for etype, count in sorted(type_counts.items()):
            st.markdown(f"- {etype}: {count}")


def render_graph_view(graph: KnowledgeGraph, selected_types: list[str], max_nodes: int):
    st.subheader("Interactive Graph Visualization")

    nodes, edges = create_agraph_data(
        graph,
        filter_types=selected_types if selected_types else None,
        max_nodes=max_nodes,
        selected_entity_id=st.session_state.selected_entity,
    )

    if not nodes:
        st.warning("No entities match the current filters.")
        return

    st.info(f"Displaying {len(nodes)} entities and {len(edges)} relationships")

    config = Config(
        width="100%",
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        node={"labelProperty": "label", "fontSize": 12},
        link={"labelProperty": "label", "renderLabel": True, "fontSize": 8},
    )

    selected = agraph(nodes=nodes, edges=edges, config=config)

    if selected:
        st.session_state.selected_entity = selected
        details = get_entity_details(graph, selected)

        if details:
            render_entity_detail_card(details)


def render_entity_detail_card(details: dict):
    entity = details["entity"]

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### 📌 {entity.name}")
        st.markdown(f"**Type:** {entity.entity_type}")
        st.markdown(f"**Description:** {entity.description or 'N/A'}")

    with col2:
        if details["outgoing"]:
            st.markdown("**Outgoing Relationships:**")
            for rel in details["outgoing"][:5]:
                st.markdown(f"- → [{rel['type']}] → {rel['target']}")

        if details["incoming"]:
            st.markdown("**Incoming Relationships:**")
            for rel in details["incoming"][:5]:
                st.markdown(f"- {rel['source']} → [{rel['type']}] →")


def render_query_view():
    st.subheader("Query the Knowledge Graph")

    if not st.session_state.rag:
        st.warning("Load a graph first to enable querying.")
        return

    query = st.text_input(
        "Enter your question",
        placeholder="What is the relationship between X and Y?",
    )

    if st.button("Ask", type="primary"):
        if not query:
            st.warning("Please enter a question.")
            return

        with st.spinner("Generating response..."):
            try:
                response = st.session_state.rag.query(query)
                st.markdown("### Response")
                st.markdown(response)
            except Exception as e:
                st.error(f"Error: {e}")


def render_entity_list(graph: KnowledgeGraph, selected_types: list[str]):
    st.subheader("Entity List")

    search = st.text_input("Search entities", placeholder="Type to filter...")

    filtered = [
        entity
        for entity in graph.entities.values()
        if (not selected_types or entity.entity_type in selected_types)
        and (not search or search.lower() in entity.name.lower())
    ]
    filtered.sort(key=lambda x: x.name)

    st.info(f"Found {len(filtered)} entities")

    display_limit = 50
    for entity in filtered[:display_limit]:
        with st.expander(f"{entity.name} [{entity.entity_type}]"):
            st.markdown(f"**Description:** {entity.description or 'N/A'}")

            rels = graph.get_relationships_for_entity(entity.entity_id)
            if rels:
                st.markdown("**Relationships:**")
                for rel in rels[:10]:
                    if rel.source_entity_id == entity.entity_id:
                        target = graph.get_entity(rel.target_entity_id)
                        if target:
                            st.markdown(
                                f"- → [{rel.relationship_type}] → {target.name}"
                            )
                    else:
                        source = graph.get_entity(rel.source_entity_id)
                        if source:
                            st.markdown(
                                f"- {source.name} → [{rel.relationship_type}] →"
                            )

    if len(filtered) > display_limit:
        st.info(
            f"Showing first {display_limit} of {len(filtered)} entities. Use search to filter."
        )


def render_welcome_screen():
    st.info(
        """
    ### Getting Started
    
    1. **Load a graph**: Enter the path to your knowledge graph JSON file in the sidebar and click "Load Graph"
    
    2. **Explore**: Use the Graph View tab to visualize and interact with the knowledge graph
    
    3. **Query**: Use the Query tab to ask questions about the knowledge graph
    
    4. **Filter**: Use the sidebar options to filter by entity type and limit the number of displayed nodes
    
    ---
    
    **Tip**: If you don't have a graph yet, process a document first using the CLI:
    ```bash
    python main.py process your_document.txt -o graph.json
    ```
    """
    )


def init_session_state():
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "selected_entity" not in st.session_state:
        st.session_state.selected_entity = None


def main():
    st.set_page_config(
        page_title="Mini-Graph-RAG Visualizer",
        page_icon="🔗",
        layout="wide",
    )

    st.title("🔗 Mini-Graph-RAG Visualizer")
    st.markdown("Interactive knowledge graph visualization and querying")

    graph_path, load_button, selected_types, max_nodes, stats_placeholder = (
        render_sidebar()
    )

    init_session_state()

    if load_button:
        try:
            with st.spinner("Loading graph..."):
                st.session_state.graph = load_graph(graph_path)
                st.session_state.rag = GraphRAG()
                st.session_state.rag.load_graph(graph_path)
            st.success("Graph loaded successfully!")
        except FileNotFoundError:
            st.error(f"File not found: {graph_path}")
        except Exception as e:
            st.error(f"Error loading graph: {e}")

    if st.session_state.graph:
        graph = st.session_state.graph
        render_stats(stats_placeholder, graph)

        tab1, tab2, tab3 = st.tabs(["🕸️ Graph View", "🔍 Query", "📋 Entity List"])

        with tab1:
            render_graph_view(graph, selected_types, max_nodes)

        with tab2:
            render_query_view()

        with tab3:
            render_entity_list(graph, selected_types)
    else:
        render_welcome_screen()

    st.divider()
    st.markdown(
        "<div style='text-align: center; color: #888;'>Mini-Graph-RAG Visualizer | Built with Streamlit</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
