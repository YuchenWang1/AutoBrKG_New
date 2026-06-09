# agents/decomposer.py
"""
Agent: DecomposerAgent
Purpose:
  Decomposes a large, unstructured report text into smaller, topically-related chunks.
  It uses an LLM to identify main themes or topics within the text (e.g., defects related to a
  specific component) and groups the original text lines under these identified topics. This
  pre-processing step helps to create more focused and contextually relevant inputs for the
  downstream ExtractorAgent.

Input:
  - report_text_content (str): A single string containing the entire content of the report,
    with different findings separated by newlines.

Output:
  - A dictionary where keys are the topics identified by the LLM (e.g., "Main Beam Defects")
    and values are lists of the original text lines that belong to that topic.
    Example: `{"Main Beam Defects": ["line 1...", "line 2..."]}`

Note:
  The prompts are currently designed for Chinese reports. For English reports, the prompts would need to be translated
  and adapted to the corresponding terminology.
"""
import os
import json
from typing import List, Dict, Any
from llm_client import get_llm_response, parse_llm_json_response


class DecomposerAgent:
    def __init__(self, model_config_name: str):
        self.model_config_name = model_config_name
        self.topic_classification_prompt_template = """
        请分析以下桥梁检测文本：
        {context}

        请识别文本中描述的主要主题（例如：关于特定构件的病害、特定类型的检测发现等，例如“主梁病害”，“桥面铺装裂缝”，“支座问题”）。
        然后将每一行文本归类到最相关的主题下。
        最终请输出一个JSON对象，其中键是识别出的主题名称，值是一个包含该主题下所有相关原始文本行（保持原样，不要修改或省略）的列表。
        例如：
        ```json
        {{
            "主梁病害": [
                "L2#箱梁梁底左侧面锚固区混凝土，距2号墩35m处，距左边缘0m处1条露筋，长度3m。",
                "L3#箱梁梁底混凝土，2条裂缝，长度3m。"
            ],
            "桥面铺装问题": [
                "桥面铺装层可见多处横向裂缝，宽度不均。",
                "沥青面层局部推移。"
            ],
            "其他未明确主题": [
                "支座脱空10%。"
            ]
        }}
        ```
        确保所有原始文本行都被包含在某个主题下。如果某些行不属于明确的重复性主题，可以将它们归类到“其他”或一个更通用的主题下。
        只需要最终返回```json your_classification_here ``` ,不需要给我其他任何内容。
        """
        # """
        # --- ENGLISH TRANSLATION OF THE PROMPT ---
        # Please analyze the following bridge inspection text:
        # {context}
        #
        # Please identify the main topics described in the text (e.g., defects related to a specific component, findings of a specific type, such as "Main Beam Defects", "Bridge Deck Pavement Cracks", "Bearing Issues").
        # Then, classify each line of text under the most relevant topic.
        # Finally, please output a JSON object where the keys are the identified topic names, and the values are lists containing all relevant original text lines (keep them as they are, do not modify or omit) for that topic.
        # For example:
        # ```json
        # {{
        #     "Main Beam Defects": [
        #         "1 instance of reinforcement exposure with length 3m in the concrete of the left side of L2# box girder bottom, 35m from pier 2, 0m from the left edge.",
        #         "2 cracks with length 3m in the concrete of L3# box girder bottom."
        #     ],
        #     "Bridge Deck Pavement Issues": [
        #         "Multiple transverse cracks with varying widths are visible on the bridge deck pavement layer.",
        #         "Local shoving of the asphalt surface layer."
        #     ],
        #     "Other Unspecified Topics": [
        #         "Bearing void 10%."
        #     ]
        # }}
        # ```
        # Ensure that all original text lines are included under some topic. If some lines do not belong to a clear, recurring topic, you can classify them under "Other" or a more general topic.
        # Only return ```json your_classification_here ```, do not give me any other content.
        # """

    def decompose_text_by_topic(self, report_text_content: str) -> Dict[str, List[str]]:
        """
        Decomposes the report text into segments based on topics identified by an LLM.

        Args:
            report_text_content: The full text of the report as a single string.

        Returns:
            A dictionary where keys are topics and values are lists of text lines for that topic.
        """
        prompt = self.topic_classification_prompt_template.format(context=report_text_content)

        # Get the classification from the LLM
        llm_response_text = get_llm_response(self.model_config_name, prompt)

        try:
            # Parse the JSON from the LLM's response
            classified_data = parse_llm_json_response(llm_response_text)

            # Basic validation and fallback for the LLM output
            if not isinstance(classified_data, dict):
                print(f"Warning: Decomposer LLM did not return a dictionary for topics. Got: {type(classified_data)}. Falling back to single 'general' topic.")
                # If the output is not a dictionary, group all lines under a single topic to ensure the process continues.
                return {"general_topic": report_text_content.strip().split('\n')}

            # Ensure the structure of the returned dictionary is valid (values are lists of strings)
            for topic, lines in classified_data.items():
                if not isinstance(lines, list):
                    print(f"Warning: Topic '{topic}' does not have a list of lines. Fixing.")
                    classified_data[topic] = []
                else:
                    classified_data[topic] = [str(line) for line in lines if isinstance(line, (str, bytes))]

            print(f"Decomposer classified text into topics: {list(classified_data.keys())}")
            return classified_data

        except json.JSONDecodeError as e:
            # Handle JSON parsing errors by falling back to a single topic
            print(f"Error decoding JSON from Decomposer LLM response: {e}")
            print(f"LLM Response: {llm_response_text}")
            return {"error_topic_parsing": report_text_content.strip().split('\n')}
        except Exception as e:
            # Handle other unexpected errors
            print(f"Unexpected error in decompose_text_by_topic: {e}")
            print(f"LLM Response: {llm_response_text}")
            return {"unexpected_error_topic_parsing": report_text_content.strip().split('\n')}
