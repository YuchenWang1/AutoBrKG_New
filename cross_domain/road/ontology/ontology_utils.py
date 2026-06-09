###############################################################################
# road_defect_validator.py
# Adapted for the road-defect ontology.
###############################################################################
import json
from rdflib import Graph, Namespace, RDF, RDFS, XSD, OWL, URIRef, Literal, BNode
from rdflib.collection import Collection
import re
from pathlib import Path

###############################################################################
# Namespaces
###############################################################################
ONT  = Namespace("http://example.org/bridge-defect-ontology#")
INST = Namespace("http://example.org/instance/")

###############################################################################
# --- Utility functions ------------------------------------------------------
###############################################################################
def make_safe_uri_component(name: str) -> str:
    name = str(name).strip()
    for old, new in [
        ("#", "_sharp_"), (" ", "_"), ("/", "_slash_"), ("?", "_qmark_"),
        ("&", "_amp_"), (":", "_colon_"), ("%", "_percent_"),
        ("<", "_lt_"), (">", "_gt_"), ("\"", "_quot_"), ("'", "_apos_")
    ]:
        name = name.replace(old, new)
    return name

def get_or_create_instance_uri(name_str, type_from_json, g, entity_uris_map, instance_actual_types_map):
    original_name = name_str.strip()
    safe_name_key = make_safe_uri_component(original_name)
    ontology_class_name = type_from_json.strip()

    uri = entity_uris_map.get(safe_name_key)
    if not uri:
        uri = INST[safe_name_key]
        entity_uris_map[safe_name_key] = uri
        ontology_class_uri = ONT[ontology_class_name]
        g.add((uri, RDF.type, ontology_class_uri))
        instance_actual_types_map[uri] = ontology_class_uri
        g.add((uri, RDFS.label, Literal(original_name, lang="zh")))
    return uri

###############################################################################
# --- JSON to Turtle ---------------------------------------------------------
###############################################################################
def convert_json_to_ttl(json_data_str, format_error_messages_list):
    try:
        data = json.loads(json_data_str) if isinstance(json_data_str, str) else json_data_str
    except json.JSONDecodeError as e:
        format_error_messages_list.append(f"严重错误: JSON 文本无法解析 - {e}")
        return None

    entries = data if isinstance(data, list) else [data]
    g = Graph()
    g.bind("ont",  ONT)
    g.bind("inst", INST)
    g.bind("rdf",  RDF)
    g.bind("rdfs", RDFS)

    entity_uris = {}
    instance_ontological_types = {}

    for entry in entries:
        # ---------- Process triples ----------
        for triple_str in entry.get("三元组", []):
            parts = triple_str.split('>')
            if len(parts) != 3:
                format_error_messages_list.append(
                    f"格式错误: 三元组 '{triple_str}' 应为 '类型:名称>关系>类型:名称'。")
                continue
            subj_full, pred_name_str, obj_full = parts

            # Parse subject and object entity descriptors.
            try:
                subj_type, subj_name = subj_full.split(':', 1)
                obj_type, obj_name = obj_full.split(':', 1)
            except ValueError:
                format_error_messages_list.append(
                    f"格式错误: 三元组 '{triple_str}' 中实体部分缺少 ':' 分隔。")
                continue

            # Create instances and add the triple.
            subj_uri = get_or_create_instance_uri(subj_name, subj_type, g,
                                                  entity_uris, instance_ontological_types)
            obj_uri  = get_or_create_instance_uri(obj_name,  obj_type,  g,
                                                  entity_uris, instance_ontological_types)
            pred_uri = ONT[pred_name_str.strip()]
            g.add((subj_uri, pred_uri, obj_uri))

        # ---------- Process attributes ----------
        # Keep the defect -> description category -> description value pattern.
        for attr_str in entry.get("属性", []):
            parts = [s.strip() for s in attr_str.split('>')]
            if len(parts) != 3:
                format_error_messages_list.append(
                    f"格式错误: 属性 '{attr_str}' 应为 '病害名称>属性类别>属性值'。")
                continue
            dis_name, cat_name, val_name = parts

            main_key = make_safe_uri_component(dis_name)
            main_uri = entity_uris.get(main_key)

            if not main_uri:
                format_error_messages_list.append(
                    f"属性 '{attr_str}' 的病害实体 '{dis_name}' 未在三元组中声明。")
                continue

            actual_type = instance_ontological_types.get(main_uri)
            if actual_type != ONT.病害:
                tq = g.qname(actual_type) if actual_type else "未知类型"
                format_error_messages_list.append(
                    f"属性关联错误: '{dis_name}' 当前类型为 {tq}，仅支持为 ont:病害 添加属性。")
                continue

            cat_uri = get_or_create_instance_uri(
                cat_name, "病害性状描述类别", g, entity_uris, instance_ontological_types)
            val_uri = get_or_create_instance_uri(
                val_name, "病害性状数值", g, entity_uris, instance_ontological_types)

            g.add((main_uri, ONT.性状描述类别是, cat_uri))
            g.add((cat_uri,  ONT.性状数值是,  val_uri))

    return g.serialize(format="turtle")

###############################################################################
# --- Ontology rule validation -----------------------------------------------
###############################################################################
def get_entity_types(uri, graph):
    return set(graph.objects(uri, RDF.type))

def get_property_domains(prop_uri, ontology_graph):
    return set(ontology_graph.objects(prop_uri, RDFS.domain))

def get_property_ranges(prop_uri, ontology_graph):
    return set(ontology_graph.objects(prop_uri, RDFS.range))

def is_subclass_or_equivalent(child, parent, ont):
    if child == parent:  # quick check
        return True
    # rdfs:subClassOf*
    ask = """
    ASK { ?c rdfs:subClassOf* ?p . }
    """
    return bool(ont.query(ask, initBindings={'c': child, 'p': parent}))

def check_type_compatibility(actual, expected, ont):
    """
    Supports direct class URIs and anonymous owl:unionOf classes.
    """
    if isinstance(expected, URIRef):
        return is_subclass_or_equivalent(actual, expected, ont)

    if isinstance(expected, BNode):
        union_node = next(ont.objects(expected, OWL.unionOf), None)
        if union_node:
            coll = Collection(ont, union_node)
            return any(check_type_compatibility(actual, member, ont) for member in coll)

    # Other complex OWL expressions are not expanded here.
    return False

def fmt(uri):
    if isinstance(uri, URIRef):
        return uri.split('#')[-1] if '#' in uri else uri.split('/')[-1]
    return str(uri)

def fmt_set(s):
    return "{" + ", ".join(sorted(fmt(u) for u in s)) + "}"

def validate_graph(data_g: Graph, ont_g: Graph):
    issues = []

    # -- Collect property URIs declared by the ontology.
    all_props = set(ont_g.subjects(RDF.type, RDF.Property))   | \
                set(ont_g.subjects(RDF.type, OWL.ObjectProperty)) | \
                set(ont_g.subjects(RDF.type, OWL.DatatypeProperty))

    # -- Track custom properties used by each instance as subject or object.
    roles = {}  # {uri: {'as_subj': set(props), 'as_obj': set(props)}}
    for s, p, o in data_g:
        if str(p).startswith(str(ONT)):
            if isinstance(s, URIRef) and str(s).startswith(str(INST)):
                roles.setdefault(s, {'as_subj': set(), 'as_obj': set()})['as_subj'].add(p)
            if isinstance(o, URIRef) and str(o).startswith(str(INST)):
                roles.setdefault(o, {'as_subj': set(), 'as_obj': set()})['as_obj'].add(p)

        # ---------- Property declaration check ----------
        if str(p).startswith(str(ONT)) and p not in all_props:
            issues.append(f"警告：关系“{fmt(p)}”未在本体中声明。")

        # ---------- Domain and range checks ----------
        if p in all_props:
            domains = get_property_domains(p, ont_g)
            ranges  = get_property_ranges(p, ont_g)

            # Domain
            if domains:
                subj_types = get_entity_types(s, data_g)
                if subj_types and not any(check_type_compatibility(t, exp, ont_g)
                                          for t in subj_types for exp in domains):
                    issues.append(
                        f"主实体类型错误：{fmt(s)} 的类型 {fmt_set(subj_types)} 不符合 {fmt_set(domains)} 的域要求 (属性 {fmt(p)})")

            # Range
            if ranges:
                if isinstance(o, URIRef):
                    obj_types = get_entity_types(o, data_g)
                    if obj_types and not any(check_type_compatibility(t, exp, ont_g)
                                             for t in obj_types for exp in ranges):
                        issues.append(
                            f"宾实体类型错误：{fmt(o)} 的类型 {fmt_set(obj_types)} 不符合 {fmt_set(ranges)} 的值域要求 (属性 {fmt(p)})")
                elif isinstance(o, Literal):
                    # XSD datatype range check.
                    xsd_ranges = {r for r in ranges if isinstance(r, URIRef) and str(r).startswith(str(XSD))}
                    if xsd_ranges and ((o.datatype and o.datatype not in xsd_ranges) or
                                       (o.datatype is None and XSD.string not in xsd_ranges)):
                        issues.append(
                            f"字面量类型错误：字面量 '{o}' 不符合属性 {fmt(p)} 期望的数据类型 {fmt_set(xsd_ranges)}")

    # ---------- Role-constraint checks ----------
    type_role_map = {
        ONT.路面类型:     {'as_subj': [ONT.道路位置是], 'as_obj': []},
        ONT.桩号:        {'as_subj': [ONT.检查指标是, ONT.存在病害是], 'as_obj': [ONT.道路位置是]},
        ONT.检查指标:     {'as_subj': [ONT.检查指标值是], 'as_obj': [ONT.检查指标是]},
        ONT.检查指标值:   {'as_subj': [], 'as_obj': [ONT.检查指标值是]},
        ONT.病害:        {'as_subj': [ONT.性状描述类别是], 'as_obj': [ONT.存在病害是]},
        ONT.病害性状描述类别: {'as_subj': [ONT.性状数值是], 'as_obj': [ONT.性状描述类别是]},
        ONT.病害性状数值: {'as_subj': [], 'as_obj': [ONT.性状数值是]},
    }

    for ent, r in roles.items():
        types = get_entity_types(ent, data_g)
        subj_props = r['as_subj']
        obj_props  = r['as_obj']

        valid_subj = set()
        valid_obj  = set()
        for t in types:
            if t in type_role_map:
                valid_subj |= set(type_role_map[t]['as_subj'])
                valid_obj  |= set(type_role_map[t]['as_obj'])

        # Subject-role violation
        bad_subj = subj_props - valid_subj
        if bad_subj:
            issues.append(
                f"角色冲突(主语)：实体 {fmt(ent)} 类型 {fmt_set(types)} 不应作为 {fmt_set(bad_subj)} 的主语。")

        # Object-role violation
        bad_obj = obj_props - valid_obj
        if bad_obj:
            issues.append(
                f"角色冲突(宾语)：实体 {fmt(ent)} 类型 {fmt_set(types)} 不应作为 {fmt_set(bad_obj)} 的宾语。")

    return issues

###############################################################################
# --- Public entry point -----------------------------------------------------
###############################################################################
def validate_json_instance(json_input: str,
                           ontology_path: str = "ontology.ttl") -> str:
    """Validate JSON instance data and return a Chinese report string."""
    format_errors = []
    ttl_str = convert_json_to_ttl(json_input, format_errors)

    # Persist TTL for debugging.
    ttl_file = Path("instance_data.ttl")
    if ttl_str:
        ttl_file.write_text(ttl_str, encoding="utf-8")

    # Load ontology.
    ont_g = Graph()
    try:
        ont_g.parse(ontology_path, format="turtle")
    except Exception as e:
        format_errors.append(f"本体文件加载失败: {e}")

    # Load instances.
    data_g = Graph()
    if ttl_file.exists():
        try:
            data_g.parse(ttl_file.as_posix(), format="turtle")
        except Exception as e:
            format_errors.append(f"实例 TTL 解析失败: {e}")

    # --- Ontology rule validation ---
    ont_issues = []
    if not format_errors:
        ont_issues = validate_graph(data_g, ont_g)

    # --- Report aggregation ---
    lines = []
    if format_errors:
        lines.append("发现以下输入格式/解析问题：")
        lines += [f"- {msg}" for msg in format_errors]

    if ont_issues:
        if not lines:
            lines.append("校验结果：")
        lines.append("发现以下本体规则违规：")
        lines += [f"- {msg}" for msg in ont_issues]

    if not lines:
        return "✓ 未发现任何格式问题或本体规则违规，数据符合规范。"
    return "\n".join(lines)

###############################################################################
# Import validate_json_instance from other scripts when needed.
###############################################################################
