# agents/constructor.py
"""
Agent: ConstructorAgent
Purpose:
  Constructs a knowledge graph in a Neo4j database from the final, structured information.
  It converts the flat list of extracted data into a graph format, creates nodes and relationships,
  and handles data housekeeping tasks like merging duplicate nodes.

Input:
  - data (list): A list of dictionaries containing the final, cleaned text, triple, and attribute fields.
  - bridge_name (str): The name of the bridge, used to create a root node for the graph components.

Output:
  - None. The agent's output is the populated graph in the Neo4j database.

Note:
  This agent does not use LLM prompts. It interfaces directly with a Neo4j database.
  It relies on the APOC library for certain database operations like merging nodes.
"""
import json
import uuid
import re
import threading
from neo4j import GraphDatabase
import logging
from utils1.config_loader import get_model_config

logger = logging.getLogger("BridgeProcessor")


def _convert_attributes_to_triples(data: list) -> list:
    """
    Transforms the input data structure for graph creation.
    - Renames the canonical triple field to the relation-extraction result field.
    - Converts the canonical attribute list into a new characteristic-extraction list, where each attribute
      becomes a structured triple ready to be converted into a subgraph.
    """
    result = []
    if not isinstance(data, list):
        logger.warning("Input data for conversion is not a list. Returning empty.")
        return result

    for item in data:
        # Create a map of entity names to their types from the main triples.
        # This is needed to find the entity type for attributes.
        name_to_type_map = {}
        for triple in item.get("三元组", []):
            parts = triple.split(">")
            if len(parts) != 3: continue
            subject_info, _, object_info = parts
            subject_parts = subject_info.split(":")
            if len(subject_parts) == 2:
                name_to_type_map[subject_parts[1].strip()] = subject_parts[0].strip()
            object_parts = object_info.split(":")
            if len(object_parts) == 2:
                name_to_type_map[object_parts[1].strip()] = object_parts[0].strip()

        # Process the main relationships.
        relation_results = []
        for triple in item.get("三元组", []):
            parts = triple.split(">")
            if len(parts) != 3: continue
            subject_info, relation, object_info = parts
            subject_type, subject_name = subject_info.split(":", 1)
            object_type, object_name = object_info.split(":", 1)
            relation_results.append({
                "关系": relation.strip(),
                "主实体类型": subject_type.strip(),
                "宾实体类型": object_type.strip(),
                "主实体": subject_name.strip(),
                "宾实体": object_name.strip()
            })

        # Convert attributes (e.g., "Defect>length>1m") into structured trait triples.
        trait_results = []
        for attribute_str in item.get("属性", []):
            parts = attribute_str.split(">")
            if len(parts) != 3:
                logger.warning(f"Malformed attribute skipped: {attribute_str}")
                continue
            entity_name, trait_type, trait_value = parts[0].strip(), parts[1].strip(), parts[2].strip()
            entity_type = name_to_type_map.get(entity_name)
            if not entity_type:
                logger.warning(f"Could not find entity type for '{entity_name}' in attribute '{attribute_str}'. Skipping.")
                continue
            trait_results.append({
                "实体": entity_name,
                "实体类型": entity_type,
                "性状类别": trait_type,
                "性状数值": trait_value
            })

        result.append({
            "文本": item.get("文本", ""),
            "关系提取结果": relation_results,
            "性状提取结果": trait_results
        })
    return result


def _create_graph_tx(tx, data, entity_ids, bridge_name):
    """
    A single Neo4j transaction function to build graph components.
    - Creates nodes and relationships for the main entities.
    - Creates separate sub-graphs for attributes (traits) to avoid node merging.
    - Creates a main 'Bridge' node and links it to the root components.
    """
    created_component_ids = set(entity_ids.values())

    # Step 1: Process main entity relationships.
    for entry in data:
        for relation in entry.get("关系提取结果", []):
            subject_name, object_name = relation.get('主实体'), relation.get('宾实体')
            subject_id, object_id = entity_ids.get(subject_name), entity_ids.get(object_name)

            if not all([subject_name, object_name, subject_id, object_id]):
                logger.warning(f"Skipping relationship due to missing data: {relation}")
                continue
            # Use MERGE to create nodes if they don't exist or match existing ones.
            query = (
                f"MERGE (a:`{relation['主实体类型']}` {{name: $subject_name, unique_id: $subject_id}}) "
                f"MERGE (b:`{relation['宾实体类型']}` {{name: $object_name, unique_id: $object_id}}) "
                f"MERGE (a)-[r:`{relation['关系']}`]->(b)"
            )
            tx.run(query, subject_name=subject_name, subject_id=subject_id, object_name=object_name, object_id=object_id)

    # Step 2: Process traits (attributes) by creating new nodes for each.
    for entry in data:
        for trait in entry.get("性状提取结果", []):
            entity_name = trait.get('实体')
            entity_id = entity_ids.get(entity_name)
            if not entity_id:
                logger.warning(f"Cannot create trait for entity '{entity_name}' as it has no ID. Trait: {trait}")
                continue
            # Generate new UUIDs for trait nodes to ensure they are always created, not merged.
            params = {
                "entity_id": entity_id,
                "trait_type_name": trait.get('性状类别'),
                "trait_type_id": str(uuid.uuid4()),
                "trait_value_name": trait.get('性状数值'),
                "trait_value_id": str(uuid.uuid4()),
            }
            # Use CREATE for trait nodes to represent each instance of an attribute uniquely.
            query = """
            MATCH (entity {unique_id: $entity_id})
            CREATE (trait_type:性状类别 {name: $trait_type_name, unique_id: $trait_type_id})
            CREATE (trait_value:性状数值 {name: $trait_value_name, unique_id: $trait_value_id})
            CREATE (entity)-[:病害性状类别是]->(trait_type)
            CREATE (trait_type)-[:性状数值是]->(trait_value)
            """
            tx.run(query, params)

    # Step 3: Link all created root components to the main bridge node.
    if bridge_name and created_component_ids:
        tx.run("MERGE (b:Bridge {name: $bridge_name})", bridge_name=bridge_name)
        # Find nodes that are roots of a subgraph (no incoming relationships from other components)
        # and link them to the bridge.
        query = """
        MATCH (b:Bridge {name: $bridge_name})
        UNWIND $id_list AS component_id
        MATCH (c {unique_id: component_id})
        WHERE NOT (()-->(c)) AND NOT (c:性状类别) AND NOT (c:性状数值) AND NOT (c:Bridge)
        MERGE (b)-[:结构构件是]->(c)
        """
        tx.run(query, bridge_name=bridge_name, id_list=list(created_component_ids))
        logger.info(f"Bridge '{bridge_name}' linked to its components.")


def _find_and_merge_duplicate_nodes_tx(tx):
    """
    Transaction function to find and merge duplicate nodes based on label and name.
    This helps clean up the graph after initial construction.
    Requires the APOC plugin in Neo4j.
    """
    logger.info("Checking for duplicate nodes to merge (excluding trait nodes)...")
    # Get all node labels except for trait-related and Bridge labels.
    labels_query = "CALL db.labels() YIELD label WHERE NOT label IN ['性状类别', '性状数值', 'Bridge'] RETURN label"
    try:
        labels_to_check = [record["label"] for record in tx.run(labels_query)]
    except Exception as e:
        logger.error(f"Failed to query labels, cannot proceed with merging. Is APOC installed? Error: {e}")
        return

    merge_count = 0
    for label in labels_to_check:
        # Group nodes by name for each label to find duplicates.
        grouping_query = f"""
        MATCH (n:`{label}`)
        WITH n.name AS name, collect(n) AS nodes
        WHERE name IS NOT NULL AND size(nodes) > 1
        RETURN name, nodes
        """
        try:
            duplicates = tx.run(grouping_query)
            for record in duplicates:
                nodes_to_merge = record["nodes"]
                logger.info(f"Merging {len(nodes_to_merge)} duplicate nodes for '{record['name']}' with label '{label}'.")
                # Use apoc.refactor.mergeNodes to merge nodes, combining properties and relationships.
                merge_query = """
                WITH $nodes AS nodes_to_merge
                CALL apoc.refactor.mergeNodes(nodes_to_merge, {properties: 'combine', mergeRels: true})
                YIELD node
                RETURN node
                """
                tx.run(merge_query, nodes=nodes_to_merge)
                merge_count += 1
        except Exception as e:
            logger.error(f"Error merging duplicates for label '{label}': {e}")
            continue

    if merge_count > 0:
        logger.info(f"Completed merging duplicates for {merge_count} groups.")
    else:
        logger.info("No duplicate nodes found to merge.")


class ConstructorAgent:
    def __init__(self, model_config_name: str):
        self.model_config_name = model_config_name
        neo4j_config = get_model_config("neo4j_config")
        if not neo4j_config:
            error_msg = "Neo4j configuration ('neo4j_config') not found in config file."
            logger.error(error_msg)
            raise ValueError(error_msg)

        NEO4J_URI = neo4j_config.get("uri")
        NEO4J_USERNAME = neo4j_config.get("username")
        NEO4J_PASSWORD = neo4j_config.get("password")

        if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
            error_msg = "Neo4j config is missing 'uri', 'username', or 'password'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.lock = threading.Lock()

    def close(self):
        """Closes the Neo4j database driver connection."""
        self.driver.close()

    def clear_database(self):
        """Clears all nodes and relationships from the Neo4j database."""
        logger.info("Clearing the Neo4j database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared.")

    def construct_graph(self, data: list, bridge_name: str):
        """
        Main method to convert data and build the graph in Neo4j.
        """
        if not data:
            logger.warning("Constructor received no data to build. Skipping graph construction.")
            return
        if not bridge_name:
            logger.warning("Constructor called without a bridge_name. Bridge will not be created.")

        logger.info("Converting raw data to graph-structured format...")
        graph_data = _convert_attributes_to_triples(data)

        entity_ids = {}
        # Pre-assign unique IDs to all main entities to ensure they can be consistently
        # referenced and merged across different data entries within the same transaction.
        with self.lock:
            for entry in graph_data:
                for relation in entry.get("关系提取结果", []):
                    for entity_key in ['主实体', '宾实体']:
                        entity_name = relation.get(entity_key)
                        if entity_name and entity_name not in entity_ids:
                            entity_ids[entity_name] = str(uuid.uuid4())

        logger.info(f"Beginning graph construction for bridge: {bridge_name}")
        with self.driver.session(database="neo4j") as session:
            # Execute the graph creation and duplicate merging in separate transactions.
            session.execute_write(_create_graph_tx, graph_data, entity_ids, bridge_name)
            logger.info(f"Completed initial graph creation for {len(graph_data)} entries.")
            session.execute_write(_find_and_merge_duplicate_nodes_tx)
            logger.info("Duplicate node check and merge complete.")
