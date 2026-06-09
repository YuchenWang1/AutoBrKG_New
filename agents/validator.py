# agents/validator.py
"""
Agent: ValidatorAgent
Purpose:
  Validates the data extracted by the ExtractorAgent against a formal ontology and a set of predefined rules.
  It assesses the quality of the extraction, provides a score, and generates feedback for the CorrectorAgent.

Input:
  - extracted_data_json_str (str): A JSON string representing the list of dictionaries of extracted data from the Extractor.
  - ontology_ttl_path (str): Path to the ontology file (e.g., in Turtle format).

Output:
  - A tuple containing:
    - feedback_json_str (str): A JSON string with validation feedback and suggested modifications.
    - score (float): A numerical score from 0.0 to 1.0 indicating the quality of the extraction.

Note:
  The prompts are currently designed for Chinese reports. For English reports, the prompts would need to be translated
  and adapted to the corresponding terminology.
"""
import json
import re
import ast
from typing import List, Dict, Optional, Any, Tuple
from llm_client import get_llm_response, parse_llm_json_response
from knowledge.knowledge_base import KnowledgeBase # For memory retrieval prompt
from utils1.ontology import validate_json_instance # Assuming ontology.py is in utils

class ValidatorAgent:
    def __init__(self, model_config_name: str, kb_json_path: str, ontology_ttl_path: str):
        self.model_config_name = model_config_name
        self.knowledge_base = KnowledgeBase(kb_json_path)
        self.ontology_ttl_path = ontology_ttl_path


        self.combined_check_fusion_prompt_template = """
                请你整合本体检查结果和提取内容检查结果，进行综合评分和提出修改建议。
                本体检查结果如下：
                {ontology_results}

                提取内容（待检查）如下：
                {extraction_context}

                作为参考的知识库示例如下（如果提取内容与此类似且本体无误，直接给1分，不需要进行下述检查【待修改部分】可输出 "无"）：
                {sample_example}

                请遵循以下规则进行处理：
                1- 针对“提取内容”进行分析和评价，务必参考“本体检查结果”和“知识库示例”。
                2- 首先，判断“知识库示例”是否和“提取内容”中的文本高度类似。如果非常类似且“本体检查结果”无明显错误，则评分可趋近1.0分，【待修改部分】可输出 "无"。
                3- 如果不类似或“本体检查结果”指出了问题，请进行详细检查：
                    a) 格式检查：确保“提取内容”的JSON格式正确，特别是//三元组//和//属性//部分。
                    b) 是否缺少病害位置信息，例如，缺少病害位置信息（例如：文本包含病害位置信息，但是模型提取后缺少病害位置信息，如”距0#台处1.5m，距左侧人行道4m处“、“2-4#、2-5#梁间”、“锚固区”、“3#梁处”、“左侧边缘跨中处”等）
                    c) 是否缺少构件部位（例如：文本包含构件部位信息，但是模型提取后缺少部位信息，如“路桥连接处”、“左侧非机动车道”、“台后搭板路桥连接处”、“台后搭板”、“右侧路缘石”、“左侧机动车道锚固区混凝土”等）
                    d) 构件提取错误（例如，文本“右侧绿化带路缘石第4跨处左侧边缘跨中处1块装饰面砖脱落”，提取的构件应该是“绿化带”，构件部位应该是“右侧路缘石”）
                    e) 构件检查：一旦“索塔平台”、“湿接缝”、“横梁”、“梯道”、“梁”、“侧墙”、“绿化带”、“人行道”内容出现在文本中，一定为”构件“。
                    （注意：绝对不可以出现编号信息，例如“构件：2-1#梁”就是错误信息，应该修改为“构件：梁”和“构件编号：2-1#”；注意构件不可以是“翼缘板”，该信息是“构件部位”）
                    （注意文本出现“梯道”和“横梁”时，构件就该为“梯道”、“横梁”，不可以是“墩顶横梁”）
                    f) 构件编号检查：构件编号示例：“1#”、“13-2#”、“L0#”、“第一跨”。
                    （注意：如果存在提取不正确的情况，一定要修改，例如："构件:扶手>构件位置是>构件编号:3#台后右侧梯道1处",构件编号仅应该是“3#台”，“后右侧梯道”应该是构件部位）
                    （注意构件编号不可以存在方位词，例如“构件编号:第1跨左侧”应该改为“构件编号:第1跨”）
                    g) 构件部位检查：构件部位：“墩顶”、“模板”、“底板”、“腹板”、“翼缘板”、“台顶”、“台帽”、“东侧”、“西侧”、“装饰板”、“路桥连接处”、“左侧非机动车道”、“台后搭板路桥连接处”、“台后搭板”、“右侧机动车道”、“右侧路缘石”、“路缘石”
                    （注意：文本包含方位词，一定确保”构件部位“包含合适方位词，例如：”右侧翼缘板“、”右侧腹板“、”墩顶西侧面“等，不可以是”翼缘板“）
                    （注意作为”构件部位“不要出现”构件“信息，例如：“构件部位：梁底板”错误不应该包含“构件：梁”，应该为“构件部位：底板”，”构件部位:墩顶横梁西侧面“错误不应该包含“构件：横梁”，应该为”构件部位:墩顶西侧面“）“墩顶”不用删除
                    （注意：构件部位不要重复包含该文本的“构件”的信息，例如：“右侧人行道5#台后3块面砖断裂”，构件是“人行道”，但是构件部位不可以是“右侧人行道”，构件部位是“右侧”）
                    h) 病害位置检查：病害位置示例：“2-4#、2-5#梁间”、”距0#台处1.5m，距左侧人行道4m处“、“锚固区”等。
                    （注意：可能存在病害位置漏掉的情况；注意病害位置不要包含“构件部位”信息（例如：“路桥连接处”、“左侧非机动车道”、“台后搭板路桥连接处”、“台后搭板”、“右侧路缘石”），不要出现重复信息，例如：“病害位置：左侧非机动车道距左侧人行道3m，距4#墩6m处”，实际“左侧非机动车道”应该为构件部位，应该修改为：“构件部位：左侧非机动车道”、“病害位置：距左侧人行道3m，距4#墩6m处”）
                    （注意，病害位置信息不可以是单独的“左侧”、“右侧”这些内容）
                    i) 病害检查：病害示例：裂缝、剥落等，注意不要漏掉病害信息
                    j) 内容检查：
                        （1）对照“本体检查结果”，检查实体和关系是否符合桥梁专业术语和预设本体结构。
                        （2）//病害位置//描述的复杂性本身不扣分，但需确保与原始文本一致。
                        （3）构件编号的 "#" 可选。
                    k) 属性检查：
                        （1）重点检查“长度”、“宽度”、“面积”等数量词是否包含具体数据信息，例如：“纵向裂缝>宽度>Wmax”就是错误的，因为宽度后应该是具体的数字。
                        （2）重点检查原始文本中的//数量//等属性是否已提取（例如：//1处//、//1条//）。
                    l) 逻辑链路：参考“本体检查结果”中关于实体关系顺序的提示。
                4- 评分规则：
                    - 初始评分为1.0分。
                    - 每发现一项明确的、可修正的错误（如格式错误、内容与本体冲突、关键属性遗漏、逻辑链路错误），酌情扣分（例如0.1-0.2分）。
                    - 对于本体检查已指出的问题，若在提取内容中确实存在，应反映在评分和待修改部分。
                    - //病害位置//可以不存在，允许复杂和冗长，病害位置不可以拆封，例如：距3#台处4m,距左侧人行道4m处，不可以中断拆开；
                5- 输出格式必须是JSON，包含 "待修改部分" 和 "评分"。
                   如果无错误：
                   ```json
                   {{
                     "待修改部分": "无",
                     "评分": 1.0
                   }}
                   ```
                   如果存在错误（请尽量具体到哪条文本或哪个部分，如果适用）：
                   ```json
                   {{
                     "待修改部分": {{
                       "文本一的简短标识或索引": ["针对文本一的问题1描述", "针对文本一的问题2描述"],
                       "general_issues": ["适用于整体的修改建议1"]
                     }},
                     "评分": 0.7
                   }}
                   ```
                6- 【待修改部分】应清晰指出问题所在和修改方向。确保所有原始文本行都被考虑到。
                只需要最终返回```json your_evaluation_and_fusion_results_here ``` ,不需要给我其他任何内容。
                """

    def _generate_adaptive_prompt_for_validation(self, extracted_data_json_str: str) -> str:
        """
        Generates an memory retrieval prompt by searching for a similar example in the knowledge base.
        If found, it returns the similar example. Otherwise, it returns a default set of examples.
        """
        text_content_for_search = ""
        try:
            # Parse the JSON string to get the text for searching
            extracted_list = json.loads(extracted_data_json_str)
            if isinstance(extracted_list, list) and extracted_list:
                first_item = extracted_list[0]
                if isinstance(first_item, dict):
                    text_content_for_search = first_item.get("文本", "")
        except (json.JSONDecodeError, TypeError, IndexError) as e:
            print(f"Warning: Could not parse text for validation's memory retrieval prompt from: {extracted_data_json_str[:100]}. Error: {e}")
            text_content_for_search = extracted_data_json_str[:200]

        # If no text is found for searching, return a generic, hardcoded prompt
        if not text_content_for_search:
            return """一定参考下述示例，包含基本的提取规则：
            [
            {"文本": "L3#台处伸缩缝锚固区混凝土1条纵向裂缝，l=0.3m，W=0.15mm",
            "三元组": ["构件:伸缩缝>构件位置是>构件编号:L3#台",
                      "构件编号:L3#台>病害具体位置是>病害位置:锚固区混凝土",
                      "病害位置:锚固区混凝土>存在病害是>病害:纵向裂缝"],
            "属性": ["纵向裂缝>数量>1条",
                     "纵向裂缝>长度>0.3m",
                     "纵向裂缝>宽度>0.15mm"]
            },
            {"文本": "4-13#铰缝1处局部脱落，L=5m",
            "三元组": ["构件:铰缝>构件位置是>构件编号:4-13#",
                      "构件编号:4-13#>存在病害是>病害:脱落"],
            "属性": ["脱落>数量>1处",
                    "脱落>长度>5m"]
            },
            {"文本": "第2跨右侧装饰板外侧面1处破损",
            "三元组": ["构件:装饰板>构件位置是>构件编号:第2跨",
                      "构件编号:第2跨>具体部位是>构件部位:右侧外侧面",
                      "构件部位:右侧外侧面>存在病害是>病害:破损"],
            "属性": ["破损>数量>1处"]
            },
            {"文本": "L2#箱梁梁底左侧面锚固区混凝土，距2号墩35m处，距左边缘0m处1条露筋，长度3m。",
            "三元组": ["构件:箱梁>构件位置是>构件编号:L2#",
                      "构件编号:L2#>具体部位是>构件部位:梁底左侧面锚固区混凝土",
                      "构件部位:梁底左侧面锚固区混凝土>病害具体位置是>病害位置:距2号墩35m处，距左边缘0m处",
                      "病害位置:距2号墩35m处，距左边缘0m处>存在病害是>病害:露筋"],
            "属性": ["露筋>数量>1条",
                     "露筋>长度>3m"]
            },
            {"文本": "1#盖梁东侧3#梁处竖向裂缝1条，L=1.2m，W=0.11mm",
            "三元组": ["构件:盖梁>构件位置是>构件编号:1#",
                      "构件编号:1#>具体部位是>构件部位:东侧",
                      "构件部位:东侧>病害具体位置是>病害位置:3#梁处",
                      "病害位置:3#梁处>存在病害是>病害:竖向裂缝"],
            "属性": ["竖向裂缝>数量>1条",
                    "竖向裂缝>长度>1.2m",
                    "竖向裂缝>宽度>0.11mm"]}
            ]"""
        # Search for a similar example in the knowledge base
        similar_example = self.knowledge_base.search_similar(text_content_for_search)
        if similar_example:
            # If found, format it as a JSON string
            return json.dumps({
                "文本": similar_example.get("文本"),
                "三元组": similar_example.get("三元组"),
                "属性": similar_example.get("属性")
            }, ensure_ascii=False, indent=2)
        # If no similar example is found
        return "无（知识库中未找到类似示例）"

    def validate_and_fuse_extraction(self, extracted_data_json_str: str) -> Tuple[str, float]:
        """
        Validates the extracted data using ontology and LLM, then fuses results and provides a score.
        Args:
            extracted_data_json_str: JSON string of a list of dicts from Extractor.
        Returns:
            A tuple containing the JSON string of validation/fusion feedback and the score.
        """
        # Step 1: Perform ontology check
        ontology_issues_str = validate_json_instance(extracted_data_json_str, self.ontology_ttl_path)
        print(f"--------------------------\nValidator - Ontology Check Results:\n{ontology_issues_str}")

        # Step 2: Generate an memory retrieval sample for the LLM validation prompt
        adaptive_sample_for_llm_val = self._generate_adaptive_prompt_for_validation(extracted_data_json_str)

        # Step 3: Construct the full prompt for the LLM check and fusion
        llm_check_fusion_prompt = self.combined_check_fusion_prompt_template.format(
            ontology_results=ontology_issues_str if ontology_issues_str.strip() else "本体检查无明显问题。",
            extraction_context=extracted_data_json_str,
            sample_example=adaptive_sample_for_llm_val
        )

        # Step 4: Get the response from the LLM
        llm_response_str = get_llm_response(self.model_config_name, llm_check_fusion_prompt)
        parsed_llm_output = parse_llm_json_response(llm_response_str)

        # Step 5: Parse the LLM output to get the score and feedback
        score = 0.0
        # Default feedback if LLM parsing fails or returns an unexpected structure
        feedback_json_str = json.dumps({"待修改部分": "LLM解析失败或未返回标准格式的反馈和评分", "评分": 0.0}, ensure_ascii=False)

        if isinstance(parsed_llm_output, dict):
            score = float(parsed_llm_output.get("评分", 0.0))
            if "待修改部分" not in parsed_llm_output:
                # Ensure the key exists, even if LLM omits it when the score is perfect
                parsed_llm_output["待修改部分"] = "无" if score >= 1.0 else "LLM未提供具体的待修改部分"
            feedback_json_str = json.dumps(parsed_llm_output, ensure_ascii=False)
        elif isinstance(parsed_llm_output, list) and parsed_llm_output: # Handle if LLM wraps in a list by mistake
            first_item = parsed_llm_output[0]
            if isinstance(first_item, dict):
                score = float(first_item.get("评分", 0.0))
                if "待修改部分" not in first_item:
                    first_item["待修改部分"] = "无" if score >= 1.0 else "LLM未提供具体的待修改部分"
                feedback_json_str = json.dumps(first_item, ensure_ascii=False)
        else:
            print(f"Warning: Validator's LLM did not return a dict as expected. LLM Response Snippet: {llm_response_str[:300]}. Parsed as: {str(parsed_llm_output)[:300]}")

        print(f"Validator - LLM Check & Fusion Score: {score}, Feedback (JSON): {feedback_json_str}")
        return feedback_json_str, score
