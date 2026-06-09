# agents/extractor.py
"""
Agent: ExtractorAgent
Purpose:
  Extracts structured information (entities, relations, and attributes) from raw text segments.
  It uses a combination of Retrieval-Augmented Generation (RAG) to pull relevant information
  from a document store and few-shot examples from a knowledge base to guide the LLM's extraction process.

Input:
  - text_to_extract (str): A string containing one or more lines of text from a bridge inspection report.

Output:
  - A list of dictionaries, where each dictionary represents a structured extraction from a line of text.
    Each dictionary contains the canonical text, triple, and attribute fields used by the pipeline.
    Example: a dictionary with text, triple-list, and attribute-list entries.

Note:
  The prompts are currently designed for Chinese reports. For English reports, the prompts would need to be translated
  and adapted to the corresponding terminology.
"""
import json
import re
from typing import List, Dict, Optional, Any, Tuple
from llm_client import get_llm_response, parse_llm_json_response
from knowledge.knowledge_base import KnowledgeBase
from knowledge.rag_utils import retrieve_relevant_chunks


class ExtractorAgent:
    def __init__(self, model_config_name: str, kb_json_path: str, rag_pdf_path: str):
        self.model_config_name = model_config_name
        self.knowledge_base = KnowledgeBase(kb_json_path)
        self.rag_pdf_path = rag_pdf_path

        self.sys_prompt = "您是桥梁领域专家。您根据提供的上下文，保证提取结果的实体符合桥梁领域知识答案。"
        self.base_prompt_template = """
        IMPORTANT: The "文本" field in your JSON output MUST be an exact copy of the input sentence provided in {context}.
        请按照下面的步骤进行，完成对：{context}内的每一行的桥梁检测文本的实体、关系、属性提取任务：
        RAG 辅助信息: 以下是从相关文档中检索到的信息，可能对当前提取有帮助：
           {rag_context}
        1- 根据给出的8个实体//构件编号（例如：1#、13-2#、L0#、第一跨）、构件（例如：湿接缝、横梁）、构件部位（例如：墩顶、模板、底板、腹板、翼缘板、台顶、台帽、路桥连接处、左侧非机动车道、台后搭板路桥连接处、台后搭板、右侧路缘石）、病害位置（例如：距0#台处1.5m，距左侧人行道4m处、锚固区等）、病害、病害数量（例如：3条等）、病害性状描述类别（例如：宽度、长度、面积等）、病害性状数值（例如：3厘米、3.45平方米等）//进行实体识别；
        2- 根据给出的4个关系//构件位置是（构件到构件编号的关系)、具体部位是（构件编号到构件部位的关系）、病害具体位置是（构件部位到病害位置的关系）、存在病害是（病害位置到病害的关系）//进行关系识别
        3- //病害数量、病害性状描述类别、病害性状数值//3个实体为//病害//实体的属性，例如：（病害：数量：病害数量，病害性状描述类别：病害性状数值）
        4- 属性检查：对于//最大长度、最大宽度、裂缝宽度、总长度、总宽度，总面积//全部修改为//长度、宽度、面积//，删除全部修饰词，数量词仅可以作为属性；
        5- 提取示例：可以对比是否和下述例子类似，如果类似则参考下述例子的提取规则。例子：{sample_adaptive_prompt}
        6- 请按照给定输出样式，输出实体关系提取格式：```json 
        [{{
        "文本": "L3#台处伸缩缝锚固区混凝土1条纵向裂缝，l=0.3m，W=0.15mm",
        "三元组": ["构件:伸缩缝>构件位置是>构件编号:L3#台",
                  "构件编号:L3#台>病害具体位置是>病害位置:锚固区混凝土",
                  "病害位置:锚固区混凝土>存在病害是>病害:纵向裂缝"],
        "属性": ["纵向裂缝>数量>1条",
                "纵向裂缝>长度>0.3m",
                "纵向裂缝>宽度>0.15mm"]
        }},
        {{
        "文本": "3#支座脱空15%",
        "三元组": ["构件:支座>构件位置是>构件编号:3#", 
                  "构件编号:3#>存在病害是>病害:脱空"],
        "属性": ["脱空>脱空率>15%"]
        }},
                {{
        "文本": "第2跨右侧装饰板外侧面1处破损",
        "三元组": [
            "构件:装饰板>构件位置是>构件编号:第2跨",
            "构件编号:第2跨>具体部位是>构件部位:右侧外侧面",
            "构件部位:右侧外侧面>存在病害是>病害:破损"
        ],
        "属性": [
            "破损>数量>1处"
        ]
        }},
                {{
        "文本": "L2#箱梁梁底左侧面锚固区混凝土，距2号墩35m处，距左边缘0m处1条露筋，长度3m。",
        "三元组": [
            "构件:箱梁>构件位置是>构件编号:L2#",
            "构件编号:L2#>具体部位是>构件部位:梁底左侧面锚固区混凝土",
            "构件部位:梁底左侧面锚固区混凝土>病害具体位置是>病害位置:距2号墩35m处，距左边缘0m处",
            "病害位置:距2号墩35m处，距左边缘0m处>存在病害是>病害:露筋"
        ],
        "属性": [
            "露筋>数量>1条",
            "露筋>长度>3m"
        ]
        }}
        ] 
        ```
        7- 完成句子提取,请注意json格式的正确性，多条数据时最外层应该包含中括号
        只需要最终返回```json your_extraction_here ``` ,不需要给我其他任何内容。
        """
        # """
        # --- ENGLISH TRANSLATION OF THE PROMPT ---
        # IMPORTANT: The "Text" field in your JSON output MUST be an exact copy of the input sentence provided in {context}.
        # Please follow the steps below to complete the entity, relation, and attribute extraction task for each line of the bridge inspection text within {context}:
        # RAG Auxiliary Information: The following information has been retrieved from relevant documents and may be helpful for the current extraction:
        #    {rag_context}
        # 1- Perform entity recognition based on the 8 given entity types: //Component ID (e.g., 1#, 13-2#, L0#, First Span), Component (e.g., Wet Joint, Crossbeam), Component Part (e.g., Pier Top, Formwork, Bottom Slab, Web Plate, Wing Plate, Abutment Top, Abutment Cap, bridge-road connection, left non-motorized lane, expansion plate at bridge-road connection, expansion plate, right curb), Defect Location (e.g., 1.5m from abutment 0#, 4m from left sidewalk, anchorage zone, etc.), Defect, Defect Quantity (e.g., 3 strips), Defect Characteristic Type (e.g., width, length, area), Defect Characteristic Value (e.g., 3 cm, 3.45 sqm)//.
        # 2- Perform relation recognition based on the 4 given relation types: //is located at (relation from Component to Component ID), has part (relation from Component ID to Component Part), has defect at (relation from Component Part to Defect Location), has defect (relation from Defect Location to Defect)//.
        # 3- The 3 entities //Defect Quantity, Defect Characteristic Type, Defect Characteristic Value// are attributes of the //Defect// entity, e.g., (Defect: quantity: Defect Quantity, Defect Characteristic Type: Defect Characteristic Value).
        # 4- Attribute Check: For //max length, max width, crack width, total length, total width, total area//, change all to //length, width, area//. Remove all modifiers. Quantitative words can only be attributes.
        # 5- Extraction Sample: You can compare if the text is similar to the example below. If so, refer to its extraction rules. Example: {sample_adaptive_prompt}
        # 6- Please follow the given output style. Output the entity-relation extraction format as: ```json
        # [{{
        # "Text": "1 longitudinal crack in expansion joint anchorage zone concrete at abutment L3#, l=0.3m, W=0.15mm",
        # "Triples": ["Component:Expansion Joint>is located at>Component ID:L3# Abutment",
        #           "Component ID:L3# Abutment>has defect at>Defect Location:Anchorage zone concrete",
        #           "Defect Location:Anchorage zone concrete>has defect>Defect:Longitudinal crack"],
        # "Attributes": ["Longitudinal crack>quantity>1 strip",
        #                "Longitudinal crack>length>0.3m",
        #                "Longitudinal crack>width>0.15mm"]
        # }},
        # {{
        # "Text": "3# bearing void 15%",
        # "Triples": ["Component:Bearing>is located at>Component ID:3#",
        #           "Component ID:3#>has defect>Defect:Void"],
        # "Attributes": ["Void>void ratio>15%"]
        # }},
        # {{
        # "Text": "1 instance of damage on the outer surface of the right decorative panel of the 2nd span",
        # "Triples": [
        #     "Component:Decorative Panel>is located at>Component ID:2nd Span",
        #     "Component ID:2nd Span>has part>Component Part:Right outer surface",
        #     "Component Part:Right outer surface>has defect>Defect:Damage"
        # ],
        # "Attributes": [
        #     "Damage>quantity>1 place"
        # ]
        # }},
        # {{
        # "Text": "1 instance of reinforcement exposure with length 3m in the concrete of the left side of L2# box girder bottom, 35m from pier 2, 0m from the left edge.",
        # "Triples": [
        #     "Component:Box Girder>is located at>Component ID:L2#",
        #     "Component ID:L2#>has part>Component Part:Concrete on the left side of beam bottom anchorage zone",
        #     "Component Part:Concrete on the left side of beam bottom anchorage zone>has defect at>Defect Location:35m from pier 2, 0m from the left edge",
        #     "Defect Location:35m from pier 2, 0m from the left edge>has defect>Defect:Reinforcement exposure"
        # ],
        # "Attributes": [
        #     "Reinforcement exposure>quantity>1 strip",
        #     "Reinforcement exposure>length>3m"
        # ]
        # }}
        # ]
        # ```
        # 7- Complete the sentence extraction. Please ensure the JSON format is correct; multiple data items should be enclosed in an outer bracket [].
        # Only return ```json your_extraction_here ```, do not give me any other content.
        # """

    def _generate_adaptive_prompt_example(self, text_to_extract: str) -> str:
        """
        Finds a similar example in the knowledge base to use as a few-shot prompt.
        """
        similar_example = self.knowledge_base.search_similar(text_to_extract)
        if similar_example:
            example_content = {
                "文本": similar_example.get("文本"),
                "三元组": similar_example.get("三元组"),
                "属性": similar_example.get("属性")
            }
            return json.dumps(example_content, ensure_ascii=False, indent=2)
        return "无（知识库中未找到类似示例）"

    def extract_information(self, text_to_extract: str) -> List[Dict[str, Any]]:
        """
        Extracts entities, relations, and attributes from the given text.
        Returns a list of dictionaries, where each dict represents one extracted item.
        """
        # Step 1: Retrieve relevant context using RAG from a document store.
        rag_context_str = retrieve_relevant_chunks(text_to_extract, self.rag_pdf_path, top_k=2)

        # Step 2: Generate an adaptive few-shot example from the Knowledge Base.
        adaptive_prompt_example_str = self._generate_adaptive_prompt_example(text_to_extract)
        print(adaptive_prompt_example_str)

        # Step 3: Construct the final, comprehensive prompt for the LLM.
        final_prompt = self.base_prompt_template.format(
            context=text_to_extract,
            sample_adaptive_prompt=adaptive_prompt_example_str,
            rag_context=rag_context_str
        )

        # Step 4: Call the LLM to perform the extraction.
        llm_response_text = get_llm_response(
            model_config_name=self.model_config_name,
            prompt=final_prompt,
            system_prompt=self.sys_prompt
        )

        # Step 5: Parse the JSON response from the LLM.
        extracted_data = parse_llm_json_response(llm_response_text)

        # Standardize the output to always be a list of dictionaries.
        # This handles cases where the LLM might return a single dict for a single input line.
        if isinstance(extracted_data, dict) and "文本" in extracted_data:
            extracted_data = [extracted_data]
        elif not isinstance(extracted_data, list):
            print(f"Warning: Extractor did not return a list. Got: {type(extracted_data)}. Wrapping in list.")
            if isinstance(extracted_data, dict):
                extracted_data = [extracted_data]
            else:
                # Create a placeholder with an error message if extraction fails completely.
                extracted_data = [{"文本": text_to_extract, "三元组": [], "属性": [],
                                   "error": "Extraction failed or unexpected format"}]

        # Final check to ensure each item in the list has the required keys.
        # This prevents errors in downstream agents.
        final_output_list = []
        for item in extracted_data:
            if isinstance(item, dict):
                if "文本" not in item: item["文本"] = text_to_extract
                if "三元组" not in item: item["三元组"] = []
                if "属性" not in item: item["属性"] = []
                final_output_list.append(item)
            else:
                final_output_list.append({
                    "文本": text_to_extract,
                    "三元组": [],
                    "属性": [],
                    "error": f"Invalid item format in extraction: {item}"
                })

        return final_output_list
