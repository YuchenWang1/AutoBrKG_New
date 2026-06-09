# agents/corrector.py
"""
Agent: CorrectorAgent
Purpose:
  Corrects the initial information extraction based on the feedback provided by the ValidatorAgent.
  It takes the original extraction and the validation feedback, then uses an LLM to apply the
  suggested modifications, producing a refined version of the extracted data.

Input:
  - original_extraction_json_str (str): A JSON string of the list of dicts from the Extractor.
  - fusion_feedback_json_str (str): A JSON string of the feedback from the Validator,
    including suggested modifications.

Output:
  - A JSON string representing the corrected list of extracted data dictionaries.

Note:
  The prompts are currently designed for Chinese reports. For English reports, the prompts would need to be translated
  and adapted to the corresponding terminology.
"""
import json
from typing import Dict, Any, List
from llm_client import get_llm_response, parse_llm_json_response


class CorrectorAgent:
    def __init__(self, model_config_name: str):
        self.model_config_name = model_config_name
        self.correction_prompt_template = """
        根据{context}内的“提取结果”按照“待修改部分”的文本和问题进行修改，输出提取结果：
        1- 不允许修改“文本”内容，不允许修改任何行文格式，例如：“:”、“>”
        2- 检查修改要求是否合理，是否符合原文本内容，对于“病害描述复杂”的建议拒绝修改
        3- 根据“待修改部分”问题，根据“待修改部分”的“问题”修改“提取结果”
        4- 构件、构件编号、构件部位、病害位置、病害，这5个实体是依次，第一个一定是构件，最后一个一定是病害，其余内容允许不存在
        5- 对于属性要求增加类别，请先检查原文是否包含这类属性；要求减少属性，拒绝修改
        5- 请按照给定输出样式，输出格式：```json 
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
        6- 确保修改后符合信息提取的要求：
            根据给出的8个实体//构件编号（例如：1#、13-2#、L0#、第一跨）、构件（例如：湿接缝、横梁）、构件部位（例如：墩顶、模板、底板、腹板、翼缘板、台顶、台帽）、病害位置（例如：距0#台处1.5m，距左侧人行道4m处、锚固区等）、病害、病害数量（例如：3条等）、病害性状描述类别（例如：宽度、长度、面积等）、病害性状数值（例如：3厘米、3.45平方米等）//进行实体识别；
            根据给出的4个关系//构件位置是（构件到构件编号的关系)、具体部位是（构件编号到构件部位的关系）、病害具体位置是（构件部位到病害位置的关系）、存在病害是（病害位置到病害的关系）//进行关系识别
            //病害数量、病害性状描述类别、病害性状数值//3个实体为//病害//实体的属性，例如：（病害：数量：病害数量，病害性状描述类别：病害性状数值）           
        7- 完成句子修改,请注意json格式的正确性，多条数据时最外层应该包含中括号
        请注意，请拒绝需要删除原文本中信息的要求，即待提取信息的“文本”是绝对不可以改变的。
        只需要最终返回```json your_modification_here ``` ,不需要给我其他任何内容。
        """
        # """
        # --- ENGLISH TRANSLATION OF THE PROMPT ---
        # Based on the "Extraction Result" and the issues described in the "Modification Suggestions" within {context}, modify and output the corrected extraction result:
        # 2- Check if the modification requests are reasonable and consistent with the original text content. Reject suggestions to modify for "complex defect descriptions".
        # 3- Modify the "Extraction Result" according to the "issues" listed in the "Modification Suggestions".
        # 4- The five entities: Component, Component ID, Component Part, Defect Location, Defect, must appear in sequence. The first must be Component, the last must be Defect. Intermediate entities are optional.
        # 5- If an attribute category is requested to be added, first check if the original text contains such an attribute. Reject requests to remove attributes.
        # 6- Please follow the given output style. Output format: ```json
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
        # 7- Ensure the modified result meets the information extraction requirements:
        #     - Perform entity recognition based on the 8 given entity types: //Component ID (e.g., 1#, 13-2#, L0#, First Span), Component (e.g., Wet Joint, Crossbeam), Component Part (e.g., Pier Top, Formwork, Bottom Slab, Web Plate, Wing Plate, Abutment Top, Abutment Cap), Defect Location (e.g., 1.5m from abutment 0#, 4m from left sidewalk, anchorage zone, etc.), Defect, Defect Quantity (e.g., 3 strips), Defect Characteristic Type (e.g., width, length, area), Defect Characteristic Value (e.g., 3 cm, 3.45 sqm)//.
        #     - Perform relation recognition based on the 4 given relation types: //is located at (relation from Component to Component ID), has part (relation from Component ID to Component Part), has defect at (relation from Component Part to Defect Location), has defect (relation from Defect Location to Defect)//.
        #     - The 3 entities //Defect Quantity, Defect Characteristic Type, Defect Characteristic Value// are attributes of the //Defect// entity, e.g., (Defect: quantity: Defect Quantity, Defect Characteristic Type: Defect Characteristic Value).
        # 8- Complete the sentence modification. Please ensure the JSON format is correct; multiple data items should be enclosed in an outer bracket [].
        # Please note, reject any request that requires deleting information from the original text. The "Text" field of the information to be extracted is absolutely unchangeable.
        # Only return ```json your_modification_here ```, do not give me any other content.
        # """

    def correct_extraction(self, original_extraction_json_str: str, fusion_feedback_json_str: str) -> str:
        """
        Corrects the extraction based on feedback from the Validator.

        Args:
            original_extraction_json_str: JSON string of the list of dicts from Extractor.
            fusion_feedback_json_str: JSON string of the feedback from Validator.

        Returns:
            A JSON string of the corrected list of dictionaries.
        """
        try:
            # Combine the original extraction and the feedback into a single context object
            # for the LLM prompt. This provides the LLM with both the data to be corrected
            # and the instructions on how to correct it.
            context_for_llm = {
                "提取结果": json.loads(original_extraction_json_str),
                **json.loads(fusion_feedback_json_str)
            }
            context_str_for_prompt = json.dumps(context_for_llm, ensure_ascii=False)

        except json.JSONDecodeError as e:
            print(f"Error preparing context for corrector: {e}")
            # If context preparation fails, return the original extraction to avoid breaking the pipeline.
            return original_extraction_json_str

        # Format the prompt with the combined context
        prompt = self.correction_prompt_template.format(context=context_str_for_prompt)

        # Get the corrected data from the LLM
        llm_response = get_llm_response(self.model_config_name, prompt)
        corrected_data_json = parse_llm_json_response(llm_response)

        # Ensure the output is a list of dictionaries as expected by downstream agents.
        # LLMs can sometimes return a single dictionary instead of a list with one item.
        if not isinstance(corrected_data_json, list):
            if isinstance(corrected_data_json, dict) and "文本" in corrected_data_json:
                # Wrap a single returned item in a list
                corrected_data_json = [corrected_data_json]
            else:
                # If the format is unexpected, log the issue and fall back to the original data.
                print(f"Corrector did not return a list. Fallback to original. Got: {corrected_data_json}")
                return original_extraction_json_str

        return json.dumps(corrected_data_json, ensure_ascii=False)
