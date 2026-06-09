# agents/reviewer.py
"""
Agent: ReviewerAgent
Purpose:
  Reviews the structured data (post-correction) for logical consistency, linguistic correctness, and structural redundancy.
  It transforms each structured data item into a single, flat string for efficient LLM review and then parses the
  LLM's corrected string back into the original structured format. This helps in identifying and fixing subtle errors
  like redundant information or illogical relationship chains.

Input:
  - constructed_data_json_str (str): A JSON string representing a list of fully structured data items.

Output:
  - A JSON string representing the list of reviewed and potentially modified data items.

Note:
  The prompts are currently designed for Chinese reports. For English reports, the prompts would need to be translated
  and adapted to the corresponding terminology.
"""

import json
import re
from typing import List, Dict, Any, Optional
from llm_client import get_llm_response, parse_llm_json_response
import logging

logger = logging.getLogger("BridgeProcessor")


class ReviewerAgent:
    """
    The ReviewerAgent class is responsible for checking the extracted results.
    It transforms data into a single string for LLM review and parses it back.
    """

    def __init__(self, model_config_name: str):
        """
        Initializes the ReviewerAgent.
        Args:
            model_config_name (str): The configuration name for the LLM model.
        """
        self.model_config_name = model_config_name
        self.known_relationships = ["构件位置是", "具体部位是", "病害具体位置是", "存在病害是"]

        self.review_single_string_prompt_template = """
        您是一位专业的桥梁检测报告高级专家。请审查以下单行字符串表示的文本信息。
        该字符串的结构是：实体1:值1>关系1>实体2:值2>...>病害实体:病害值>属性1类型>属性1值>属性2类型>属性2值...
        原始文本为："{context_single_string}"
        
        审查规则：
        1. 提取结果检查：
            需要首先满足实体提取规范
            构件编号（例如：1#、13-2#、L0#、第一跨）注意，构件编号为“台处”、”内“，删除该内容和前面的关系
            构件（例如：梁、湿接缝）如果是“构件：梁底板”，则需要改为“构件：梁”和“构件部位：底板”
            构件部位（例如：模板、底板、腹板、翼缘板、台顶、台帽、横梁）注意，构件部位为“台处”、”内“，删除该内容和前面的关系
            病害位置（例如：距0#台处1.5m，距左侧人行道4m处、锚固区等）注意。病害位置为”（无方位词）“这种带括号的内容，删除括号及其内部文字，如果为空，则删除该病害位置和前面的关系
            病害 （封缝脱落）注意“露筋>数量>4处>面积>0.2㎡>破损>数量>4处>面积>0.2㎡“应该合并病害 修改为”露筋破损>数量>4处>面积>0.2㎡“，两个病害的情况，如果是”剥落“、”露筋“，则合并为”剥落露筋“；注意 病害中存在的的顿号删除
            病害数量（例如：3条等）注意，数量后面如果是数字应该加上“处”字，例如：“锈蚀>数量>1”应该修改为“锈蚀>数量>1处”
            病害性状描述类别（例如：宽度、长度、面积等）、病害性状数值（例如：3厘米、3.45平方米等）//进行实体识别；
            构件部位为墩顶，修改为墩顶处，例如“构件编号:4#墩>具体部位是>构件部位:墩顶”修改为“构件编号:4#墩>具体部位是>构件部位:墩顶处”
        2. 链路冗余检查：
            重复或类型的内容出现（例如：伸缩缝和伸缩缝内、主梁和梁、盖梁和梁、西侧和西侧、东侧和东侧、排水系统和排水系统部位、箱梁和箱梁箱），删除第二个和前面的关系。
           - 示例：如果输入字符串为 "构件:桥面铺装>构件位置是>构件编号:L1#>具体部位是>构件部位:桥面铺装>存在病害是>病害:裂缝"
             其中 "构件部位:桥面铺装" 与 "构件:桥面铺装" 重复，则应删除 ">具体部位是>构件部位:桥面铺装" 部分。
             修正后应为： "构件:桥面铺装>构件位置是>构件编号:L1#>存在病害是>病害:裂缝"
           - 示例：如果输入字符串为 "构件:伸缩缝>构件位置是>构件编号:0#台>病害具体位置是>病害位置:伸缩缝内>存在病害是>病害:沉积物阻塞>数量>1处"
             其中 "构件:伸缩缝" 与 "病害位置:伸缩缝内" 重复，则应删除 ">病害具体位置是>病害位置:伸缩缝内" 部分。
             修正后应为： "构件:桥面铺装>构件位置是>构件编号:L1#>存在病害是>病害:裂缝"
           - 注意：当“构件部位”与“构件”的实体值类似即可才视为冗余，例如“构件:桥面铺装”和“构件部位:桥面铺装”、“构件:伸缩缝”和“构件部位:伸缩缝内”。如果“构件:箱梁”和“构件部位:梁底左侧”，则不是冗余。
        3. 构件部位冗余检查：
            如果构件部位出现了和构件重复的内容，则删除构件部位中的构件内容。
           - 示例：如果输入字符串为 "构件:支座>构件位置是>构件编号:1-0-9#>具体部位是>构件部位:支座钢垫板>存在病害是>病害:锈蚀"
             其中 "构件部位:支座钢垫板" 与 "构件:支座" 重复，则应删除 "构件部位:支座钢垫板" 中的 "支座" 修正真正的构件部位为 "构件部位:钢垫板" 
             修正后应为： "构件:支座>构件位置是>构件编号:1-0-9#>具体部位是>构件部位:钢垫板>存在病害是>病害:锈蚀"
           - 注意：构件部位冗余与链路冗余检查不冲突
           - 示例：如果输入字符串为 "构件:铰缝>构件位置是>构件编号:4-19#>具体部位是>构件部位:铰缝1处>存在病害是>病害:脱落>数量>1处>长度>20m"
             其中 "构件:铰缝" 与 "构件部位:铰缝1处" 重复，则应删除 ">具体部位是>构件部位:铰缝1处" 
             修正后应为： "构件:铰缝>构件位置是>构件编号:4-19#>存在病害是>病害:脱落>数量>1处>长度>20m"
        3. 病害冗余检查：
            如果病害出现了类似（例如：纵向裂缝和裂缝、裂缝位置和裂缝），则删除第一个病害和前面的关系，然后将第二个病害修改为更全面的名字
           - 示例：如果输入字符串为 "构件:横梁>构件位置是>构件编号:1#墩顶>具体部位是>构件部位:西侧面2-4#、2-5#梁间>病害具体位置是>病害位置:竖向裂缝>存在病害是>病害:裂缝"
             其中 "构件部位:纵向裂缝" 与 "病害:裂缝" 重复，则应删除第一个和前面的关系 ">病害具体位置是>病害位置:竖向裂缝" 部分。修正真正的病害为“纵向裂缝”
             修正后应为： "构件:横梁>构件位置是>构件编号:1#墩顶>具体部位是>构件部位:西侧面2-4#、2-5#梁间>存在病害是>病害:裂缝"
           - 注意：病害冗余与链路冗余检查不冲突。
        4. 属性冗余检查：
            病害后面的内容如果不是长度、宽度、面积、数量等数量名词以及性状描述、病害原因等关键属性，则删除。如果是"最大长度、最大宽度、总面积"等，则删除修饰词仅保留数量名词
           - 示例：如果三元组部分已经明确了病害位置 (例如 "构件:桥面铺装>病害具体位置是>病害位置:距3#台处4m>存在病害是>病害:裂缝>长度>1m>位置>距3#台处4m")
             并且属性部分出现重复非数量描述 (例如 "位置>距3#台处4m")，
             则应从属性中删除该位置描述，即删除 ">位置>距3#台处4m" 部分。
             修正后应为： "构件:桥面铺装>病害具体位置是>病害位置:距3#台处4m>存在病害是>病害:裂缝>长度>1m"
           - 示例：如果三元组部分已经包含面积(例如 "构件:桥面铺装>病害具体位置是>病害位置:距3#台处4m>存在病害是>病害:裂缝>面积>0.25×0.2m²×3>总面积>0.15m²") 
             并且属性部分出现重复描述 (例如 ">总面积>0.15m²")，
             则应从属性中删除该位置描述，即删除 ">总面积>0.15m²" 部分。
             修正后应为： "构件:桥面铺装>病害具体位置是>病害位置:距3#台处4m>存在病害是>病害:裂缝>面积>0.25×0.2m²×3"
           - 示例：如果出现数量名词包含“最大、总体、累计”等修饰词，删除修饰词(例如 "构件:桥面铺装>病害具体位置是>病害位置:距3#台处4m>存在病害是>病害:裂缝>最大宽度>0.15m") 
             则应从属性中删除最大，即删除 "最大" 部分。
             修正后应为： "构件:桥面铺装>病害具体位置是>病害位置:距3#台处4m>存在病害是>病害:裂缝>宽度>0.15m"
        5. 实体内容检查：
            出现“病害为:裂缝1处”、“病害为:1处”等类似的描述在“病害位置”或“构件部位”时，删除该实体和前面的关系
           - 示例：如果输入字符串为(例如："构件:桥台>构件位置是>构件编号:0#>具体部位是>构件部位:台顶>病害具体位置是>病害位置:1处>存在病害是>病害:砂石堆积>数量>1处")    
             其中的“病害位置:1处” 为违法内容，则应从删除该内容和前面的关系，即删除 ">病害具体位置是>病害位置:1处" 部分。
             修正后应为： "构件:桥台>构件位置是>构件编号:0#>具体部位是>构件部位:台顶>存在病害是>病害:砂石堆积>数量>1处"     
        6. 确保实体类型（如“构件”、“构件编号”、“病害”）和关系（如“构件位置是”、“存在病害是”）的正确使用和顺序。
           请参考以下基本链路结构：
           【构件】-构件位置是-【构件编号】-具体部位是-【构件部位】-病害具体位置是-【病害位置】-存在病害是-【病害】
           此结构后面可以跟随多个 ">属性类型>属性值" 对。根据实际情况，中间的某些实体和关系对可能不存在。
           如果链路不符合上述结构（例如实体缺失或关系错误），请根据规则进行修正。
           例如，若出现 "...>构件编号:第5跨>构件位置是>构件:人行道>..." 中 “构件位置是”关系使用错误，应修正关系或实体顺序。
        
        请输出一个JSON对象，包含修正后的字符串。该JSON对象应该仅包含 "corrected_sentence" 键。
        ```json
        {{
          "corrected_sentence": "修正后的单行字符串"
        }}
        如果无需修改，请返回原始字符串在 "corrected_sentence" 字段中。
        务必只返回上述 JSON 结构，不要包含任何其他解释性文字或前缀。
        """
        # """
        # --- ENGLISH TRANSLATION OF THE PROMPT ---
        # You are a senior expert in bridge inspection reporting. Please review the following single-line string representation of text information.
        # The string structure is: Entity1:Value1>Relation1>Entity2:Value2>...>DefectEntity:DefectValue>Attribute1Type>Attribute1Value>Attribute2Type>Attribute2Value...
        # The original text is: "{context_single_string}"
        #
        # Review Rules:
        # 1. Extraction Result Check:
        #     - Must first comply with entity extraction specifications.
        #     - Component ID (e.g., 1#, 13-2#, L0#, First Span): Note, if the component ID is "at abutment", "inside", delete this content and the preceding relation.
        #     - Component (e.g., Beam, Wet Joint): If it is "Component: Beam Bottom Slab", it needs to be changed to "Component: Beam" and "Component Part: Bottom Slab".
        #     - Component Part (e.g., Formwork, Bottom Slab, Web Plate, Wing Plate, Abutment Top, Abutment Cap, Crossbeam): Note, if the component part is "at abutment", "inside", delete this content and the preceding relation.
        #     - Defect Location (e.g., 1.5m from abutment 0#, 4m from left sidewalk, anchorage zone, etc.): Note, if the defect location is content in parentheses like "(no directional word)", delete the parentheses and their content. If it becomes empty, delete this defect location and the preceding relation.
        #     - Defect (e.g., Sealant Spalling): Note, "Reinforcement exposure>quantity>4 places>area>0.2㎡>damage>quantity>4 places>area>0.2㎡" should be merged into "Reinforcement exposure damage>quantity>4 places>area>0.2㎡". If there are two defects like "spalling" and "reinforcement exposure", merge them into "spalling and reinforcement exposure"; also remove any enumeration commas within the defect name.
        #     - Defect Quantity (e.g., 3 strips): Note, if the quantity is just a number, add "places" (or equivalent unit), e.g., "rust>quantity>1" should be modified to "rust>quantity>1 place".
        #     - Defect characteristic description type (e.g., width, length, area), and defect characteristic value (e.g., 3 cm, 3.45 sqm) // perform entity recognition.
        #     - If a Component Part is "Pier Top", change it to "at Pier Top", e.g., "Component ID:4# Pier>has part>Component Part:Pier Top" should be modified to "Component ID:4# Pier>has part>Component Part:at Pier Top".
        # 2. Link Redundancy Check:
        #     - If repetitive or similar content appears (e.g., Expansion Joint and inside Expansion Joint, Main Beam and Beam, Cap Beam and Beam, West side and West side, East side and East side, Drainage System and Drainage System Part, Box Girder and Box Girder Box), delete the second occurrence and its preceding relation.
        #    - Example: If the input string is "Component:Bridge Deck Pavement>is located at>Component ID:L1#>has part>Component Part:Bridge Deck Pavement>has defect>Defect:Crack"
        #      Here, "Component Part:Bridge Deck Pavement" is redundant with "Component:Bridge Deck Pavement", so the ">has part>Component Part:Bridge Deck Pavement" part should be deleted.
        #      Corrected: "Component:Bridge Deck Pavement>is located at>Component ID:L1#>has defect>Defect:Crack"
        #    - Example: If the input string is "Component:Expansion Joint>is located at>Component ID:0# Abutment>has defect at>Defect Location:inside Expansion Joint>has defect>Defect:Sediment Clogging>quantity>1 place"
        #      Here, "Defect Location:inside Expansion Joint" is redundant with "Component:Expansion Joint", so the ">has defect at>Defect Location:inside Expansion Joint" part should be deleted.
        #      Corrected: "Component:Bridge Deck Pavement>is located at>Component ID:L1#>has defect>Defect:Crack"
        #    - Note: Redundancy is considered only when the entity values of "Component Part" and "Component" are similar, e.g., "Component:Bridge Deck Pavement" and "Component Part:Bridge Deck Pavement". If it's "Component:Box Girder" and "Component Part:Left side of beam bottom", it is not redundant.
        # 3. Component Part Redundancy Check:
        #     - If the component part contains content that is repetitive of the component, remove the repetitive content from the component part.
        #    - Example: If the input string is "Component:Bearing>is located at>Component ID:1-0-9#>has part>Component Part:Bearing Steel Plate>has defect>Defect:Rust"
        #      Here, "Component Part:Bearing Steel Plate" is redundant with "Component:Bearing", so "Bearing" should be removed from "Component Part:Bearing Steel Plate", making the true component part "Component Part:Steel Plate".
        #      Corrected: "Component:Bearing>is located at>Component ID:1-0-9#>has part>Component Part:Steel Plate>has defect>Defect:Rust"
        #    - Note: This check does not conflict with the link redundancy check.
        #    - Example: If the input string is "Component:Hinge Joint>is located at>Component ID:4-19#>has part>Component Part:Hinge Joint 1 place>has defect>Defect:Spalling>quantity>1 place>length>20m"
        #      Here, "Component Part:Hinge Joint 1 place" is redundant with "Component:Hinge Joint", so ">has part>Component Part:Hinge Joint 1 place" should be deleted.
        #      Corrected: "Component:Hinge Joint>is located at>Component ID:4-19#>has defect>Defect:Spalling>quantity>1 place>length>20m"
        # 4. Defect Redundancy Check:
        #     - If similar defects appear (e.g., Longitudinal Crack and Crack), delete the first defect and its preceding relation, then update the second defect to the more comprehensive name.
        #    - Example: If the input string is "Component:Crossbeam>is located at>Component ID:1# Pier Top>has part>Component Part:West face between beams 2-4#, 2-5#>has defect at>Defect Location:Vertical Crack>has defect>Defect:Crack"
        #      Here, "Defect Location:Vertical Crack" is redundant with "Defect:Crack". The first one and its relation ">has defect at>Defect Location:Vertical Crack" should be deleted. The true defect should be corrected to "Vertical Crack".
        #      Corrected: "Component:Crossbeam>is located at>Component ID:1# Pier Top>has part>Component Part:West face between beams 2-4#, 2-5#>has defect>Defect:Vertical Crack"
        #    - Note: This check does not conflict with the link redundancy check.
        # 5. Attribute Redundancy Check:
        #     - Content following the defect that is not a key attribute like length, width, area, quantity, characteristic description, or defect cause should be deleted. If attributes are "max length, max width, total area", remove the modifiers to keep only the base noun.
        #    - Example: If the triple part already specifies the defect location (e.g., "Component:Bridge Deck Pavement>has defect at>Defect Location:4m from abutment 3#>has defect>Defect:Crack>length>1m>location>4m from abutment 3#")
        #      and the attribute part has a repetitive non-quantitative description (e.g., "location>4m from abutment 3#"),
        #      then this location description should be removed from the attributes, i.e., delete ">location>4m from abutment 3#".
        #      Corrected: "Component:Bridge Deck Pavement>has defect at>Defect Location:4m from abutment 3#>has defect>Defect:Crack>length>1m"
        #    - Example: If the triple part already contains area (e.g., "Component:Bridge Deck Pavement>has defect at>Defect Location:4m from abutment 3#>has defect>Defect:Crack>area>0.25×0.2m²×3>total area>0.15m²")
        #      and the attribute part has a repetitive description (e.g., ">total area>0.15m²"),
        #      then delete this from the attributes.
        #      Corrected: "Component:Bridge Deck Pavement>has defect at>Defect Location:4m from abutment 3#>has defect>Defect:Crack>area>0.25×0.2m²×3"
        #    - Example: If a quantitative noun includes modifiers like "max, total, cumulative" (e.g., "Component:Bridge Deck Pavement>... >has defect>Defect:Crack>max width>0.15m"),
        #      remove the modifier.
        #      Corrected: "Component:Bridge Deck Pavement>... >has defect>Defect:Crack>width>0.15m"
        # 6. Entity Content Check:
        #     - If descriptions like "Defect is: 1 crack" or "Defect is: 1 place" appear in "Defect Location" or "Component Part", delete that entity and its preceding relation.
        #    - Example: If the input is ("Component:Abutment>is located at>Component ID:0#>has part>Component Part:Abutment Top>has defect at>Defect Location:1 place>has defect>Defect:Gravel Accumulation>quantity>1 place")
        #      The "Defect Location:1 place" is invalid content. Delete it and the preceding relation.
        #      Corrected: "Component:Abutment>is located at>Component ID:0#>has part>Component Part:Abutment Top>has defect>Defect:Gravel Accumulation>quantity>1 place"
        # 7. Ensure correct usage and order of entity types (like "Component", "Component ID", "Defect") and relations (like "is located at", "has defect").
        #    Please refer to the following basic link structure:
        #    [Component]-is located at-[Component ID]-has part-[Component Part]-has defect at-[Defect Location]-has defect-[Defect]
        #    This structure can be followed by multiple ">AttributeType>AttributeValue" pairs. Some intermediate entity-relation pairs may not exist depending on the case.
        #    If the chain does not conform to the above structure (e.g., missing entity or wrong relation), please correct it according to the rules.
        #    For example, if "...>Component ID:5th Span>is located at>Component:Sidewalk>..." occurs, the "is located at" relation is used incorrectly and should be corrected.
        #
        # Please output a JSON object containing the corrected string. The JSON object should only contain the "corrected_sentence" key.
        # ```json
        # {{
        #   "corrected_sentence": "The corrected single-line string"
        # }}
        # If no modifications are needed, please return the original string in the "corrected_sentence" field.
        # You must only return the JSON structure above, without any other explanatory text or prefixes.
        # """

    def _transform_item_to_single_string(self, item: Dict[str, Any]) -> str:
        """
        Transforms a structured dictionary item into a single ">" separated string.
        This flattened format is easier for the LLM to process for redundancy checks.
        """
        triples = item.get("三元组", [])
        attributes = item.get("属性", [])
        path_parts = []
        current_path_tip = None

        if not triples and not attributes: return ""

        for i, triple_str in enumerate(triples):
            parts = triple_str.split('>')
            if len(parts) != 3:
                logger.warning(f"Malformed triple in '三元组' skipped: {triple_str}")
                continue
            s, r, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
            # If this is the first triple, add all parts.
            if not path_parts:
                path_parts.extend([s, r, o])
                current_path_tip = o
            # If the subject of the current triple matches the object of the last one, chain them.
            elif current_path_tip == s:
                path_parts.extend([r, o])
                current_path_tip = o
            # If it's a disconnected triple, append it fully.
            else:
                logger.warning(
                    f"Disconnected triple in '三元组': {triple_str}. Previous tip: {current_path_tip}. Appending.")
                path_parts.extend([s, r, o])
                current_path_tip = o

        for attr_like_str in attributes:
            parts = attr_like_str.split('>')

            if len(parts) == 3 and parts[1].strip() in self.known_relationships:
                s, r, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if not path_parts:
                    path_parts.extend([s, r, o])
                    current_path_tip = o
                elif current_path_tip == s:
                    path_parts.extend([r, o])
                    current_path_tip = o
                else:
                    logger.warning(
                        f"Treating entry in '属性' as a new triple chain: {attr_like_str}. Previous tip: {current_path_tip}")
                    path_parts.extend([s, r, o])
                    current_path_tip = o
            # This handles standard attributes like "Defect>Attribute>Value".
            elif len(parts) == 3:
                # If the attribute's subject matches the last entity, just add attribute_type>attribute_value.
                if current_path_tip == parts[0].strip():
                    path_parts.extend([parts[1].strip(), parts[2].strip()])
                else:
                    # If it doesn't match, this might be a context issue, but we append for now.
                    path_parts.extend([parts[1].strip(), parts[2].strip()])
            # This handles older formats or errors where attributes are just Type>Value.
            elif len(parts) == 2:
                path_parts.extend([parts[0].strip(), parts[1].strip()])
            else:
                logger.warning(f"Malformed entry in '属性' skipped: {attr_like_str}")

        return ">".join(path_parts)

    def _parse_corrected_string_to_json(self, corrected_string: str, original_text: str) -> Dict[str, Any]:
        """
        Parses the LLM's corrected single string back into the desired structured JSON format.
        It reconstructs the canonical triple and attribute lists.
        """
        if not corrected_string:
            return {"文本": original_text, "三元组": [], "属性": []}

        parts = [p.strip() for p in corrected_string.split('>') if p.strip()]

        triples_list = []
        attributes_list = []

        # This list will store the flat sequence of entities and relations from the chain
        # e.g., [S1, R1, O1, R2, O2, R3, O3_defect]
        chain_parts_flat = []

        # Find the split point between the main relationship chain and the attributes
        attributes_start_idx = len(parts)
        idx = 0
        while idx < len(parts):
            # The first triple must be S-R-O where R is a known relationship
            if not chain_parts_flat:
                if idx + 2 < len(parts) and parts[idx + 1] in self.known_relationships:
                    chain_parts_flat.extend([parts[idx], parts[idx + 1], parts[idx + 2]])
                    idx += 3
                else: # Cannot form a valid first triple
                    attributes_start_idx = idx
                    break
            # Subsequent parts must be R-O pairs
            else:
                if idx + 1 < len(parts) and parts[idx] in self.known_relationships:
                    chain_parts_flat.extend([parts[idx], parts[idx + 1]])
                    idx += 2
                else: # The chain of relationships ends here
                    attributes_start_idx = idx
                    break

        last_disease_entity_value = None
        if len(chain_parts_flat) >= 3:
            s, r, o = chain_parts_flat[0], chain_parts_flat[1], chain_parts_flat[2]
            triples_list.append(f"{s}>{r}>{o}")
            if r == "存在病害是" and o.startswith("病害:"):
                last_disease_entity_value = o.split(":", 1)[1].strip() if ":" in o else o.strip()

            # Create subsequent triples from the chain
            for i in range(3, len(chain_parts_flat) - 1, 2):
                s_chained = chain_parts_flat[i - 1] # Previous object is the new subject
                r_chained = chain_parts_flat[i]
                o_chained = chain_parts_flat[i + 1]
                triples_list.append(f"{s_chained}>{r_chained}>{o_chained}")
                if r_chained == "存在病害是" and o_chained.startswith("病害:"):
                    last_disease_entity_value = o_chained.split(":", 1)[1].strip() if ":" in o_chained else o_chained.strip()

        # Fallback to find the last defect entity if not already found
        if not last_disease_entity_value and triples_list:
            for triple_str in reversed(triples_list):
                obj_part = triple_str.split(">")[2]
                if obj_part.startswith("病害:"):
                    last_disease_entity_value = obj_part.split(":", 1)[1].strip() if ":" in obj_part else obj_part.strip()
                    break

        # Parse attributes from the remaining parts of the string
        attr_idx = attributes_start_idx
        while attr_idx + 1 < len(parts):  # Attributes are Name>Value pairs
            attr_name = parts[attr_idx]
            attr_value = parts[attr_idx + 1]

            if last_disease_entity_value:
                # Format as "DefectName>AttributeName>AttributeValue"
                attributes_list.append(f"{last_disease_entity_value}>{attr_name}>{attr_value}")
            else:
                # Fallback: if no defect entity is found, attach to the last entity in the chain
                if triples_list:
                    fallback_entity_full = triples_list[-1].split(">")[2]
                    fallback_entity_value = fallback_entity_full.split(":", 1)[1].strip() if ":" in fallback_entity_full else fallback_entity_full.strip()
                    if fallback_entity_value:
                        logger.warning(
                            f"No specific disease entity found for attributes. Attaching '{attr_name}>{attr_value}' to last triple object: '{fallback_entity_value}'.")
                        attributes_list.append(f"{fallback_entity_value}>{attr_name}>{attr_value}")
                    else:
                        logger.error(f"Cannot determine entity for attribute '{attr_name}>{attr_value}'. Skipping.")
                else:
                    logger.error(
                        f"No triples found, cannot determine entity for attribute '{attr_name}>{attr_value}'. Skipping.")
            attr_idx += 2
        # Log any leftover parts that couldn't be parsed
        if attr_idx < len(parts):
            logger.warning(f"Orphaned attribute part(s) at end of string skipped: {'>'.join(parts[attr_idx:])}")

        return {"文本": original_text, "三元组": triples_list, "属性": attributes_list}

    def review_constructed_data(self, constructed_data_json_str: str) -> str:
            """
            Main method to review a batch of constructed data items.
            It iterates through each item, transforms it, gets LLM feedback,
            parses the result, and aggregates the reviewed items.
            """
            try:
                items_from_constructor = json.loads(constructed_data_json_str)
                if not isinstance(items_from_constructor, list):
                    logger.error(f"Reviewer expected a list, got {type(items_from_constructor)}")
                    return json.dumps([{"error": "Invalid input: not a list"}], ensure_ascii=False)
            except json.JSONDecodeError as e:
                logger.error(f"Reviewer input JSON error: {e}. Snippet: {constructed_data_json_str[:200]}")
                return json.dumps([{"error": "Invalid input JSON"}], ensure_ascii=False)

            # Flatten any nested lists to handle variations in input structure
            flattened_items: List[Any] = []
            for it in items_from_constructor:
                if isinstance(it, dict):
                    flattened_items.append(it)
                elif isinstance(it, list):
                    for sub in it:
                        if isinstance(sub, dict):
                            flattened_items.append(sub)
                        else:
                            logger.warning(f"Nested non-dict item skipped: {sub}")
                else:
                    flattened_items.append(it)
            items_from_constructor = flattened_items

            reviewed_items_list: List[Dict[str, Any]] = []
            for item_idx, item in enumerate(items_from_constructor):
                if not isinstance(item, dict):
                    reviewed_items_list.append({"error": "Invalid item type", "item_snippet": str(item)[:100]})
                    continue

                original_text = item.get("文本", "")
                if not original_text:
                    error_item = {**item, "error_review": "Missing '文本'"}
                    error_item.setdefault("文本", "UNKNOWN_TEXT")
                    error_item.setdefault("三元组", [])
                    error_item.setdefault("属性", [])
                    reviewed_items_list.append(error_item)
                    continue
                # Transform the structured item into a single string for review
                single_string_to_review = self._transform_item_to_single_string(item)
                if not single_string_to_review:
                    reviewed_items_list.append(item)
                    continue
                # Create the prompt and get the LLM's review
                prompt_for_llm = self.review_single_string_prompt_template.format(
                    context_single_string=single_string_to_review,
                    original_sentence_text=original_text
                )
                raw_llm_response_str = get_llm_response(self.model_config_name, prompt_for_llm)
                llm_output_data = parse_llm_json_response(raw_llm_response_str)

                # Extract the corrected sentence from the LLM's response
                corrected_single_string = None
                if isinstance(llm_output_data, dict):
                    corrected_single_string = llm_output_data.get("corrected_sentence")
                elif isinstance(llm_output_data, str) and ">" in llm_output_data:
                    corrected_single_string = llm_output_data

                if not corrected_single_string:
                    logger.warning(
                        f"LLM did not provide 'corrected_sentence' for: '{single_string_to_review}'. Using original.")
                    reviewed_items_list.append(item)
                    continue
                # Parse the corrected string back into the structured JSON format
                reviewed_item_dict = self._parse_corrected_string_to_json(
                    corrected_single_string, original_text
                )
                reviewed_items_list.append(reviewed_item_dict)
            # Return the final list of reviewed items as a JSON string
            return json.dumps(reviewed_items_list, ensure_ascii=False, indent=4)
