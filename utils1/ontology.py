# utils/ontology.py

"""
This module provides the core functionality for validating structured data (extracted in JSON format)
against a formal bridge defect ontology. It performs a two-stage validation process:

1.  **Format Conversion and Validation:** It first attempts to convert the input JSON data into
    RDF triples in the Turtle (TTL) format. During this conversion, it checks for basic structural
    and formatting errors in the input JSON, such as incorrect splitting of entities and relationships.

2.  **Ontological Validation:** If the conversion is successful, it loads both the generated instance
    data and the predefined ontology graph. It then validates the instance data against the ontology's
    rules, checking for:
    -   **Domain/Range Violations:** Ensures that the subjects and objects of a relationship (property)
        are of the correct type as defined by rdfs:domain and rdfs:range in the ontology.
    -   **Undefined Properties:** Warns if a relationship used in the data is not defined in the ontology.
    -   **Logical Role Conflicts:** Checks if an entity is used in a role that is inconsistent with its
        type (e.g., a "Component" entity being the object of a "hasValue" relationship).

The final output is a human-readable string detailing any format errors or ontology violations found,
or a success message if the data is valid.
"""
import json
from rdflib import Graph, Namespace, RDF, RDFS, XSD, OWL, URIRef, Literal, BNode
from rdflib.collection import Collection
import re

# Define namespaces used throughout the module, consistent with the ontology file.
ONT = Namespace("http://example.org/bridge-defect-ontology#")
INST = Namespace("http://example.org/instance/")


def make_safe_uri_component(name):
    """
    Cleans and escapes a string to make it a valid component of a URI.
    Replaces special characters that are not allowed or have special meaning in URIs.
    """
    name = str(name).strip()
    # Replace common problematic characters with safe alternatives
    name = name.replace("#", "_sharp_")
    name = name.replace(" ", "_")
    name = name.replace("/", "_slash_")
    name = name.replace("?", "_qmark_")
    name = name.replace("&", "_amp_")
    name = name.replace(":", "_colon_")
    name = name.replace("%", "_percent_")
    name = name.replace("<", "_lt_")
    name = name.replace(">", "_gt_")
    name = name.replace("\"", "_quot_")
    name = name.replace("'", "_apos_")
    return name


def get_or_create_instance_uri(name_str, type_from_json, g, entity_uris_map, instance_actual_types_map):
    """
    Retrieves an existing URI for a given entity name or creates a new one if it doesn't exist.
    This ensures that the same entity is represented by the same URI throughout the graph.
    It also adds the entity's type (rdf:type) and label (rdfs:label) to the graph upon creation.
    """
    original_name = name_str.strip()
    safe_name_key = make_safe_uri_component(original_name)
    ontology_class_name = type_from_json.strip()

    # Check if a URI for this entity already exists in our map
    uri = entity_uris_map.get(safe_name_key)
    if not uri:
        # If not, create a new URI and add it to the graph with its type and label
        uri = INST[safe_name_key]
        entity_uris_map[safe_name_key] = uri
        ontology_class_uri = ONT[ontology_class_name]
        g.add((uri, RDF.type, ontology_class_uri))
        instance_actual_types_map[uri] = ontology_class_uri
        g.add((uri, RDFS.label, Literal(original_name, lang="zh")))
    return uri


def convert_json_to_ttl(json_data_str, format_error_messages_list):
    """
    Converts a JSON string of extracted data into an RDF graph serialized in Turtle format.
    It processes the canonical triple and attribute fields from the JSON, reporting
    any format errors encountered along the way.
    """
    try:
        # Load JSON data from string or use if already a dict/list
        data = json.loads(json_data_str) if isinstance(json_data_str, str) else json_data_str
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        format_error_messages_list.append(f"Critical Error: The JSON text itself could not be parsed - {e}")
        return None

    entries = data if isinstance(data, list) else [data]
    g = Graph()
    # Bind prefixes for cleaner Turtle output
    g.bind("ont", ONT)
    g.bind("inst", INST)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)

    # Maps to keep track of created URIs and their types during conversion
    entity_uris = {}
    instance_ontological_types = {}

    for entry in entries:
        # Process the main relationship triples
        for triple_str in entry.get("三元组", []):
            try:
                # Expects format 'SubjectType:SubjectName>PredicateName>ObjectType:ObjectName'
                parts = triple_str.split('>')
                if len(parts) != 3:
                    error_msg = f"Format Error: Triple '{triple_str}' should be in 'SubjectType:SubjectName>RelationName>ObjectType:ObjectName' format. Split by '>' did not result in 3 parts (got {len(parts)})."
                    format_error_messages_list.append(error_msg)
                    continue

                subj_full_part, pred_name_str, obj_full_part = parts

                # Validate and parse subject part
                subj_parts = subj_full_part.strip().split(':', 1)
                if len(subj_parts) != 2:
                    error_msg = f"Format Error: The subject part '{subj_full_part.strip()}' of triple '{triple_str}' should be 'Type:Name'. Split by ':' did not result in 2 parts."
                    format_error_messages_list.append(error_msg)
                    continue
                subj_type_str, subj_name_str = subj_parts

                # Validate and parse object part
                obj_parts = obj_full_part.strip().split(':', 1)
                if len(obj_parts) != 2:
                    error_msg = f"Format Error: The object part '{obj_full_part.strip()}' of triple '{triple_str}' should be 'Type:Name'. Split by ':' did not result in 2 parts."
                    format_error_messages_list.append(error_msg)
                    continue
                obj_type_str, obj_name_str = obj_parts

                # Create URIs for subject, predicate, and object, and add the triple to the graph
                subj_uri = get_or_create_instance_uri(subj_name_str.strip(), subj_type_str.strip(), g, entity_uris, instance_ontological_types)
                obj_uri = get_or_create_instance_uri(obj_name_str.strip(), obj_type_str.strip(), g, entity_uris, instance_ontological_types)
                pred_uri = ONT[pred_name_str.strip()]
                g.add((subj_uri, pred_uri, obj_uri))

            except Exception as e:
                error_msg = f"System Error: An unknown error occurred while processing triple '{triple_str}' ({e}). Skipped."
                format_error_messages_list.append(error_msg)

        # Process the attributes, which are modeled as separate entities and relationships
        for attr_str in entry.get("属性", []):
            try:
                # Expects format 'EntityName>CategoryName>Value'
                parts = [s.strip() for s in attr_str.split('>')]
                if len(parts) != 3:
                    error_msg = f"Format Error: Attribute '{attr_str}' should be 'EntityName>DescriptionCategory>Value'. Parsed into {len(parts)} parts, expected 3."
                    format_error_messages_list.append(error_msg)
                    continue
                ent_name, cat_name, val_name = parts

                # Find the URI of the main entity to which the attribute belongs
                main_key = make_safe_uri_component(ent_name)
                main_uri = entity_uris.get(main_key)

                if not main_uri:
                    error_msg = f"Linking Error: The entity '{ent_name}' for attribute '{attr_str}' was not defined in any triple. Cannot link attribute. Skipped."
                    format_error_messages_list.append(error_msg)
                    continue

                # Check if the attribute is being attached to the correct entity type (e.g., a Defect)
                actual_main_type = instance_ontological_types.get(main_uri)
                if actual_main_type != ONT.病害:
                    actual_type_qname = g.qname(actual_main_type) if actual_main_type else "Unknown or undeclared type"
                    error_msg = f"Attribute Linking Error: The entity '{ent_name}' (inferred type: {actual_type_qname}) for attribute '{attr_str}' is not of type ont:病害. This implementation only supports adding attributes to defects. Skipped."
                    format_error_messages_list.append(error_msg)
                    continue

                # Create URIs for the attribute category and value, and link them to the main entity
                cat_uri = get_or_create_instance_uri(cat_name, "病害性状描述类别", g, entity_uris, instance_ontological_types)
                val_uri = get_or_create_instance_uri(val_name, "病害性状数值", g, entity_uris, instance_ontological_types)
                g.add((main_uri, ONT.具有描述类别, cat_uri)) # (Defect) -> hasDescriptionCategory -> (Category)
                g.add((cat_uri, ONT.具有数值, val_uri))       # (Category) -> hasValue -> (Value)

            except Exception as e:
                error_msg = f"System Error: An unknown error occurred while processing attribute '{attr_str}' ({e}). Skipped."
                format_error_messages_list.append(error_msg)

    # Serialize the final graph to a Turtle string
    return g.serialize(format="turtle")


def get_entity_types(entity_uri, graph):
    """Returns all rdf:type declarations for a given entity URI in a graph."""
    return set(graph.objects(entity_uri, RDF.type))


def get_property_domains(prop_uri, ontology_graph):
    """Returns all rdfs:domain definitions for a given property from the ontology graph."""
    return set(ontology_graph.objects(prop_uri, RDFS.domain))


def get_property_ranges(prop_uri, ontology_graph):
    """Returns all rdfs:range definitions for a given property from the ontology graph."""
    return set(ontology_graph.objects(prop_uri, RDFS.range))


def is_subclass_or_equivalent(class_uri, superclass_uri, ontology_graph):
    """
    Checks if class_uri is a subclass of (or the same as) superclass_uri
    using the transitive rdfs:subClassOf* property path.
    """
    if not isinstance(class_uri, URIRef) or not isinstance(superclass_uri, URIRef):
        return False
    if class_uri == superclass_uri:
        return True
    # Use SPARQL ASK query for efficient subclass checking
    q = "ASK { ?class_uri rdfs:subClassOf* ?superclass_uri . }"
    bindings = {'class_uri': class_uri, 'superclass_uri': superclass_uri}
    return bool(ontology_graph.query(q, initBindings=bindings))


def check_type_compatibility(instance_actual_type, expected_type_expression, ontology_graph):
    """
    Checks if an instance's actual type is compatible with an expected type expression
    (which can be a simple class or a complex one like a union).
    """
    if not isinstance(instance_actual_type, (URIRef, BNode)):
        return False

    # Handle complex types like owl:unionOf
    if isinstance(expected_type_expression, BNode):
        try:
            list_node = next(ontology_graph.objects(expected_type_expression, OWL.unionOf), None)
            if list_node:
                union_members = Collection(ontology_graph, list_node)
                # The instance is compatible if its type is a subclass of any member of the union
                return any(check_type_compatibility(instance_actual_type, member, ontology_graph) for member in union_members)
            return instance_actual_type == expected_type_expression
        except Exception:
            return False
    # Handle simple named class types
    elif isinstance(expected_type_expression, URIRef):
        return is_subclass_or_equivalent(instance_actual_type, expected_type_expression, ontology_graph)
    return False


def get_local_name(uri):
    """Extracts the local name from a full URI for more readable error messages."""
    if isinstance(uri, URIRef):
        uri_str = str(uri)
        if '#' in uri_str: return uri_str.split('#')[-1]
        if '/' in uri_str: return uri_str.split('/')[-1]
        return uri_str
    return str(uri)


def format_complex_type(node, ontology_graph):
    """Formats a complex BNode type (like a union) into a readable string."""
    if not isinstance(node, BNode): return get_local_name(node)
    union_list = list(ontology_graph.objects(node, OWL.unionOf))
    if union_list:
        try:
            collection = Collection(ontology_graph, union_list[0])
            return f"({' or '.join(get_local_name(member) for member in collection)})"
        except Exception as e:
            return f"UnionType[ParseError: {e}]"
    return f"ComplexType_{str(node)[-8:]}"


def format_uri_set(uri_set, ontology_graph=None):
    """Formats a set of URIs into a sorted, readable string for error messages."""
    if not uri_set: return "{}"
    readable_names = [format_complex_type(uri, ontology_graph) if isinstance(uri, BNode) and ontology_graph else get_local_name(uri) for uri in uri_set]
    return "{" + ", ".join(sorted(list(readable_names))) + "}"


def validate_graph(data_graph, ontology_graph):
    """
    Performs the main ontological validation of the instance data graph against the ontology.
    """
    issues = []
    # Get all defined properties from the ontology
    all_ontology_properties = set(ontology_graph.subjects(RDF.type, RDF.Property)) | \
                              set(ontology_graph.subjects(RDF.type, OWL.ObjectProperty)) | \
                              set(ontology_graph.subjects(RDF.type, OWL.DatatypeProperty))
    # Map to store the roles each entity plays (as subject or object)
    entity_roles = {}

    for s, p, o in data_graph:
        # Record the roles for each instance URI
        if isinstance(s, URIRef) and str(s).startswith(str(INST)):
            entity_roles.setdefault(s, {'as_subject': set(), 'as_object': set()})['as_subject'].add(p)
        if isinstance(o, URIRef) and str(o).startswith(str(INST)):
            entity_roles.setdefault(o, {'as_subject': set(), 'as_object': set()})['as_object'].add(p)

        # Check if the predicate (relationship) is defined in our ontology
        is_custom_ontology_prop = str(p).startswith(str(ONT))
        if is_custom_ontology_prop and p not in all_ontology_properties:
            issues.append(f"Warning: The relationship '{get_local_name(p)}' is not declared as a property in the ontology. Found in triple ({get_local_name(s)} > {get_local_name(p)} > {get_local_name(o)}).")

        # Perform Domain and Range checks for properties that are defined in the ontology
        if p in all_ontology_properties:
            # Check domain (subject type)
            expected_domains = get_property_domains(p, ontology_graph)
            if expected_domains and isinstance(s, URIRef) and str(s).startswith(str(INST)):
                subject_types = get_entity_types(s, data_graph)
                if not subject_types:
                    issues.append(f"Type Missing: Entity '{get_local_name(s)}' has no rdf:type, cannot validate its use as subject of '{get_local_name(p)}'.")
                elif not any(check_type_compatibility(st, ed, ontology_graph) for st in subject_types for ed in expected_domains):
                    issues.append(f"Domain Error: Entity '{get_local_name(s)}' (Type: {format_uri_set(subject_types, ontology_graph)}) as subject of '{get_local_name(p)}' violates its defined domain {format_uri_set(expected_domains, ontology_graph)}.")

            # Check range (object type)
            expected_ranges = get_property_ranges(p, ontology_graph)
            if expected_ranges:
                if isinstance(o, URIRef) and str(o).startswith(str(INST)): # If object is an instance
                    object_types = get_entity_types(o, data_graph)
                    class_ranges = {r for r in expected_ranges if not (isinstance(r, URIRef) and str(r).startswith(str(XSD)))}
                    if not object_types and class_ranges:
                        issues.append(f"Type Missing: Entity '{get_local_name(o)}' has no rdf:type, cannot validate its use as object of '{get_local_name(p)}'.")
                    elif class_ranges and not any(check_type_compatibility(ot, er, ontology_graph) for ot in object_types for er in class_ranges):
                        issues.append(f"Range Error: Entity '{get_local_name(o)}' (Type: {format_uri_set(object_types, ontology_graph)}) as object of '{get_local_name(p)}' violates its defined range {format_uri_set(class_ranges, ontology_graph)}.")
                elif isinstance(o, Literal): # If object is a literal value
                    xsd_ranges = {r for r in expected_ranges if isinstance(r, URIRef) and str(r).startswith(str(XSD))}
                    is_valid_lit = o.datatype in xsd_ranges or \
                                   (o.datatype is None and not o.language and XSD.string in xsd_ranges) or \
                                   (o.datatype is None and o.language and RDF.langString in xsd_ranges)
                    if xsd_ranges and not is_valid_lit:
                        issues.append(f"Literal Range Mismatch: Literal '{o}' (Type: {o.datatype or 'rdf:PlainLiteral'}) for property {get_local_name(p)}. Expected XSD types: {format_uri_set(xsd_ranges)}.")

    # A map defining the valid relationships each entity type can participate in (as subject or object).
    type_relationship_map = {
        ONT.病害: {'as_subject': [ONT.具有描述类别], 'as_object': [ONT.存在病害是]},
        ONT.构件: {'as_subject': [ONT.具体部位是, ONT.构件位置是, ONT.存在病害是, ONT.病害具体位置是], 'as_object': []},
        ONT.构件编号: {'as_subject': [ONT.具体部位是, ONT.存在病害是, ONT.病害具体位置是], 'as_object': [ONT.构件位置是]},
        ONT.构件部位: {'as_subject': [ONT.存在病害是, ONT.病害具体位置是], 'as_object': [ONT.具体部位是]},
        ONT.病害性状描述类别: {'as_subject': [ONT.具有数值], 'as_object': [ONT.具有描述类别]},
        ONT.病害性状数值: {'as_subject': [], 'as_object': [ONT.具有数值]},
        ONT.病害位置: {'as_subject': [ONT.存在病害是], 'as_object': [ONT.病害具体位置是]}
    }

    # Validate roles for each entity based on its type
    for entity, roles_info in entity_roles.items():
        subject_custom_props = {pr for pr in roles_info['as_subject'] if str(pr).startswith(str(ONT))}
        object_custom_props = {pr for pr in roles_info['as_object'] if str(pr).startswith(str(ONT))}
        entity_types = get_entity_types(entity, data_graph)

        if not entity_types:
            if subject_custom_props: issues.append(f"Role Warning: Untyped entity '{get_local_name(entity)}' is used as a subject. A type should be declared.")
            if object_custom_props: issues.append(f"Role Warning: Untyped entity '{get_local_name(entity)}' is used as an object. A type should be declared.")
            continue

        valid_subject_roles_for_type = set()
        valid_object_roles_for_type = set()
        for etype in entity_types:
            if etype in type_relationship_map:
                valid_subject_roles_for_type.update(type_relationship_map[etype]['as_subject'])
                valid_object_roles_for_type.update(type_relationship_map[etype]['as_object'])

        # Check for improper use as a subject
        improper_subject_props = subject_custom_props - valid_subject_roles_for_type
        if improper_subject_props:
            issues.append(f"Role Conflict (Subject): Entity '{get_local_name(entity)}' (Type: {format_uri_set(entity_types, ontology_graph)}) should not be the subject of relationships {format_uri_set(improper_subject_props)}. Allowed subject roles: {format_uri_set(valid_subject_roles_for_type) or 'None'}.")

        # Check for improper use as an object
        improper_object_props = object_custom_props - valid_object_roles_for_type
        if improper_object_props:
            issues.append(f"Role Conflict (Object): Entity '{get_local_name(entity)}' (Type: {format_uri_set(entity_types, ontology_graph)}) should not be the object of relationships {format_uri_set(improper_object_props)}. Allowed object roles: {format_uri_set(valid_object_roles_for_type) or 'None'}.")

    return issues


def validate_json_instance(json_input_str: str, ontology_path: str = "ontology.ttl") -> str:
    """
    The main entry point for validation. It orchestrates the conversion and validation process.
    """
    input_format_errors = []
    # Step 1: Convert JSON to Turtle, collecting any format errors.
    ttl_output_string = convert_json_to_ttl(json_input_str, input_format_errors)

    instance_ttl_path = "instance_data.ttl"
    if ttl_output_string:
        try:
            with open(instance_ttl_path, "w", encoding="utf-8") as f:
                f.write(ttl_output_string)
            print(f"TTL data successfully saved to {instance_ttl_path}")
        except IOError as e:
            input_format_errors.append(f"File Save Error: Could not save TTL file {instance_ttl_path}: {e}")
            ttl_output_string = None
    elif not input_format_errors:
        if json.loads(json_input_str) if isinstance(json_input_str, str) else json_input_str:
            input_format_errors.append("Note: Input JSON was valid but did not produce any serializable RDF triples.")

    ont_graph = Graph()
    data_graph = Graph()
    ontology_validation_issues = []
    can_validate_ontology = False

    # Step 2: Load the ontology graph.
    try:
        ont_graph.parse(ontology_path, format="turtle")
    except FileNotFoundError:
        input_format_errors.append(f"Ontology File Error: Ontology file '{ontology_path}' not found. Cannot perform ontology validation.")
    except Exception as e:
        input_format_errors.append(f"Ontology Load Error: Failed to load ontology '{ontology_path}': {e}. Cannot perform ontology validation.")

    # Step 3: Load the instance data graph if the ontology loaded and TTL was generated.
    if ont_graph and ttl_output_string:
        try:
            data_graph.parse(instance_ttl_path, format="turtle")
            if not data_graph and len(ttl_output_string.splitlines()) > 5:
                input_format_errors.append(f"Instance Data Parse Warning: Graph is empty after parsing '{instance_ttl_path}', but the TTL file was not empty.")
            elif data_graph:
                can_validate_ontology = True
        except Exception as e:
            input_format_errors.append(f"Instance TTL Parse Error: Failed to parse instance TTL data '{instance_ttl_path}': {e}. Cannot perform ontology validation.")

    # Step 4: Run the validation if both graphs are loaded.
    if can_validate_ontology:
        print("Starting ontology validation...")
        try:
            raw_ontology_issues = validate_graph(data_graph, ont_graph)
            ontology_validation_issues.extend(raw_ontology_issues)
            print("Ontology validation complete.")
        except Exception as e:
            ontology_validation_issues.append(f"An unexpected system error occurred during ontology validation: {e}")
            print("Ontology validation was not completed due to an error.")

    # Step 5: Compile and return the final report.
    final_report_lines = []
    if input_format_errors:
        final_report_lines.append("The following input format or processing issues were found:")
        for error_msg in input_format_errors:
            final_report_lines.append(f"- {error_msg}")

    if ontology_validation_issues:
        if final_report_lines: final_report_lines.append("") # Add a newline
        final_report_lines.append("The following ontology rule violations were found:")
        for issue_msg in ontology_validation_issues:
            final_report_lines.append(f"- {issue_msg}")

    print("Data validation process finished.")
    if not final_report_lines:
        return "No format issues or ontology rule violations were found. The instance data complies with the specification."
    else:
        return "\n".join(final_report_lines)
