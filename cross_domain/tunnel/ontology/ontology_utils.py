"""
validate_tunnel_instances.py
Validate tunnel-defect instance JSON against the tunnel-defect ontology.
Usage:
    result = validate_json_instance(json_str,
                                    ontology_path="tunnel_defect_ontology.ttl")
"""

import json, re, os
from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, URIRef, Literal, BNode
from rdflib.collection import Collection

# -----------------------------------------------------------------
#  Namespaces
# -----------------------------------------------------------------
ONT  = Namespace("http://example.org/tunnel-defect-ontology#")
INST = Namespace("http://example.org/instance/")

# -----------------------------------------------------------------
#  Utility functions
# -----------------------------------------------------------------
def make_safe_uri_component(name: str) -> str:
    name = str(name).strip()
    for bad, rep in (
        ("#", "_sharp_"), (" ", "_"), ("/", "_slash_"), ("?", "_qmark_"),
        ("&", "_amp_"), (":", "_colon_"), ("%", "_percent_"),
        ("<", "_lt_"), (">", "_gt_"), ('"', "_quot_"), ("'", "_apos_")
    ):
        name = name.replace(bad, rep)
    return name


def get_or_create_instance_uri(name_str, type_from_json, g,
                               entity_uris_map, instance_actual_types_map):
    """
    Find or create an instance URI and declare rdf:type / rdfs:label.
    """
    safe_key = make_safe_uri_component(name_str)
    uri = entity_uris_map.get(safe_key)
    if uri is None:
        uri = INST[safe_key]
        entity_uris_map[safe_key] = uri
        class_uri = ONT[type_from_json]
        g.add((uri, RDF.type, class_uri))
        g.add((uri, RDFS.label, Literal(name_str, lang="zh")))
        instance_actual_types_map[uri] = class_uri
    return uri

# -----------------------------------------------------------------
#  JSON to TTL
# -----------------------------------------------------------------
def convert_json_to_ttl(json_data_str, format_error_messages):
    """
    Convert JSON with triple/attribute fields into a TTL string.
    """
    try:
        data = json.loads(json_data_str) if isinstance(json_data_str, str) else json_data_str
    except json.JSONDecodeError as e:
        format_error_messages.append(f"严重错误: JSON 无法解析 - {e}")
        return None

    entries = data if isinstance(data, list) else [data]
    g = Graph()
    g.bind("ont", ONT)
    g.bind("inst", INST)

    entity_uris, instance_types = {}, {}

    for entry in entries:
        # ---- 3.1 Process triples -------------------------------------------
        for triple in entry.get("三元组", []):
            parts = triple.split('>')
            if len(parts) != 3:
                format_error_messages.append(
                    f"格式错误: '{triple}' 应包含 2 个 '>' 分隔符。")
                continue

            subj_desc, pred_name, obj_desc = parts
            if ':' not in subj_desc or ':' not in obj_desc:
                format_error_messages.append(
                    f"格式错误: '{triple}' 的实体描述应为 '类型:名称'。")
                continue

            subj_type, subj_name = [s.strip() for s in subj_desc.split(':', 1)]
            obj_type, obj_name = [s.strip() for s in obj_desc.split(':', 1)]

            subj_uri = get_or_create_instance_uri(
                subj_name, subj_type, g, entity_uris, instance_types
            )
            obj_uri = get_or_create_instance_uri(
                obj_name, obj_type, g, entity_uris, instance_types
            )
            pred_uri = ONT[pred_name.strip()]
            g.add((subj_uri, pred_uri, obj_uri))

        # ---- 3.2 Process attributes ----------------------------------------
        for attr in entry.get("属性", []):
            pieces = [p.strip() for p in attr.split('>')]
            if len(pieces) != 3:
                format_error_messages.append(
                    f"格式错误: 属性 '{attr}' 应为 '病害名称>描述类别>数值' 结构。")
                continue
            defect_name, cat_name, val_name = pieces

            main_uri = entity_uris.get(make_safe_uri_component(defect_name))
            if not main_uri:
                format_error_messages.append(
                    f"属性 '{attr}' 的病害实体 '{defect_name}' 未在三元组中出现，已跳过。")
                continue

            # Attributes are only allowed for defect entities.
            if instance_types.get(main_uri) != ONT.病害:
                format_error_messages.append(
                    f"属性 '{attr}' 关联的主实体 '{defect_name}' 不是 ont:病害 类型，已跳过。")
                continue

            cat_uri = get_or_create_instance_uri(
                cat_name, "病害性状描述类别", g, entity_uris, instance_types)
            val_uri = get_or_create_instance_uri(
                val_name, "病害性状数值", g, entity_uris, instance_types)

            g.add((main_uri, ONT.性状描述类别是, cat_uri))
            g.add((cat_uri, ONT.性状数值是, val_uri))

    return g.serialize(format="turtle")

# -----------------------------------------------------------------
#  Validation helpers
# -----------------------------------------------------------------
def get_entity_types(ent, graph):       return set(graph.objects(ent, RDF.type))
def get_property_domains(prop, g):      return set(g.objects(prop, RDFS.domain))
def get_property_ranges(prop, g):       return set(g.objects(prop, RDFS.range))

# Reusable subclass/union compatibility check.
def is_subclass_or_equivalent(a, b, g):
    if not all(isinstance(x, (URIRef, BNode)) for x in (a, b)): return False
    if a == b: return True
    q = "ASK { ?a rdfs:subClassOf* ?b . }"
    return bool(g.query(q, initBindings={'a': a, 'b': b}))

def check_type_compatibility(inst_type, expected, g):
    if isinstance(expected, BNode):
        lst = next(g.objects(expected, OWL.unionOf), None)
        if lst:
            return any(check_type_compatibility(inst_type, m, g)
                       for m in Collection(g, lst))
        return inst_type == expected
    return is_subclass_or_equivalent(inst_type, expected, g)

def get_local_name(uri):
    if isinstance(uri, URIRef):
        return uri.split('#')[-1] if '#' in uri else uri.split('/')[-1]
    return str(uri)

def format_uri_set(s, g=None):
    names = [get_local_name(u) for u in s]
    return "{" + ", ".join(sorted(names)) + "}"

# -----------------------------------------------------------------
#  Core validation
# -----------------------------------------------------------------
def validate_graph(data_graph: Graph, ont_graph: Graph):
    """Check domain/range constraints, role constraints, and value links."""
    issues = []

    # ---------- 1) Domain / range ----------

    # ---------- 2) Role checks ----------
    type_role = {
        ONT.构件:              {'as_subject':[ONT.构件部位是],         'as_object':[]},
        ONT.构件部位:          {'as_subject':[ONT.病害位置是],         'as_object':[ONT.构件部位是]},
        ONT.里程:              {'as_subject':[ONT.检测方法是, ONT.检测部位是, ONT.存在病害是],
                                'as_object':[ONT.病害位置是]},
        ONT.检测方法:          {'as_subject':[],                       'as_object':[ONT.检测方法是]},
        ONT.检测部位:          {'as_subject':[],                       'as_object':[ONT.检测部位是]},
        ONT.病害:              {'as_subject':[ONT.性状描述类别是],     'as_object':[ONT.存在病害是]},
        ONT.病害性状描述类别:  {'as_subject':[ONT.性状数值是],         'as_object':[ONT.性状描述类别是]},
        ONT.病害性状数值:      {'as_subject':[],                       'as_object':[ONT.性状数值是]},
    }

    # Collect custom subject/object predicates for each entity.
    roles = {}
    for s, p, o in data_graph:
        if str(p).startswith(str(ONT)):                       # Custom ontology relations only.
            if isinstance(s, URIRef) and str(s).startswith(str(INST)):
                roles.setdefault(s, {'subj': set(), 'obj': set()})['subj'].add(p)
            if isinstance(o, URIRef) and str(o).startswith(str(INST)):
                roles.setdefault(o, {'subj': set(), 'obj': set()})['obj'].add(p)

    # Compare allowed and observed predicates for each entity.
    for ent, rinfo in roles.items():
        ent_types = get_entity_types(ent, data_graph)

        # ---- Subject role ----
        if rinfo['subj']:
            allowed = {pr for t in ent_types for pr in type_role.get(t, {}).get('as_subject', [])}
            illegal = rinfo['subj'] - allowed
            if illegal:
                issues.append(
                    f"角色冲突(主语)：实体“{get_local_name(ent)}” "
                    f"不应作主语出现在 {format_uri_set(illegal)}。")

        # ---- Object role ----
        if rinfo['obj']:
            allowed = {pr for t in ent_types for pr in type_role.get(t, {}).get('as_object', [])}
            illegal = rinfo['obj'] - allowed
            if illegal:
                issues.append(
                    f"角色冲突(宾语)：实体“{get_local_name(ent)}” "
                    f"不应作宾语出现在 {format_uri_set(illegal)}。")

    # ---------- 3) Business rule: each description category needs a value ----------
    for cat in data_graph.subjects(RDF.type, ONT.病害性状描述类别):
        if not list(data_graph.objects(cat, ONT.性状数值是)):
            issues.append(
                f"缺失值：病害性状描述类别实体“{get_local_name(cat)}”未连接任何病害性状数值。")

    return issues

# -----------------------------------------------------------------
#  Public entry point
# -----------------------------------------------------------------
def validate_json_instance(json_input_str: str,
                           ontology_path: str = "ontology.ttl") -> str:
    fmt_errors = []
    ttl_str = convert_json_to_ttl(json_input_str, fmt_errors)

    ont_graph, data_graph = Graph(), Graph()
    try:
        ont_graph.parse(ontology_path, format="turtle")
    except Exception as e:
        fmt_errors.append(f"本体加载失败: {e}")

    if ttl_str:
        try:
            data_graph.parse(data=ttl_str, format="turtle")
        except Exception as e:
            fmt_errors.append(f"实例 TTL 解析失败: {e}")

    issues = []
    if not fmt_errors:
        issues = validate_graph(data_graph, ont_graph)

    # ----------- Output report -----------
    if not fmt_errors and not issues:
        return "未发现格式问题或本体规则违规，实例数据符合规范。"
    txt = []
    if fmt_errors:
        txt.append("发现以下输入/解析问题：")
        txt += [f"- {e}" for e in fmt_errors]
    if issues:
        txt.append("发现以下本体规则违规：")
        txt += [f"- {i}" for i in issues]
    return "\n".join(txt)


