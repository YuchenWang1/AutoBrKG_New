# main.py

"""
This script serves as the main orchestrator for a multi-agent data processing pipeline.
The pipeline is designed to extract structured information from bridge inspection reports,
validate and refine it, and finally load it into a Neo4j knowledge graph.

The overall workflow is as follows:
1.  **Setup**: Initializes logging with colored output for better readability.
2.  **Report Loading**: Reads a raw text file containing the inspection report. The first line
    is expected to be the bridge's name, followed by inspection findings.
3.  **Decomposition**: The `DecomposerAgent` splits the report into smaller, topically-related
    chunks of text. This helps in providing more focused context to the subsequent agents.
4.  **Parallel Processing**: A thread pool is used to process each line of the report concurrently.
    Each line undergoes an "extract-validate-correct" loop:
    a. `ExtractorAgent`: Extracts initial structured data (entities, relationships, attributes).
    b. `ValidatorAgent`: Scores the extraction against an ontology and business rules.
    c. `CorrectorAgent`: If the score is low, it attempts to correct the extraction based on feedback.
    d. This loop can run for a few iterations to improve data quality.
5.  **Knowledge Base Update**: Verified, high-quality extractions (score >= 1.0) are dynamically
    added to a JSON-based knowledge base. This allows the `ExtractorAgent` in other threads to
    retrieve newly validated examples for memory-guided few-shot prompting.
6.  **Review**: After all lines are processed, the `ReviewerAgent` performs a final, holistic check
    on the entire dataset for logical consistency and redundancy.
7.  **Sorting**: The final, reviewed data is sorted to match the original order of lines in the
    inspection report, ensuring the final output is coherent.
8.  **Graph Construction**: The `ConstructorAgent` takes the final, clean data and builds a
    knowledge graph in a Neo4j database, clearing any pre-existing data for that bridge.
"""
import os
import json
import time
import logging
from datetime import datetime
import concurrent.futures
from typing import List, Dict, Any
from threading import Lock

# Attempt to enable color support on Windows terminals
if os.name == 'nt':
    try:
        import colorama
        colorama.init()
    except ImportError:
        pass

# --- Agent Imports ---
from agents.decomposer import DecomposerAgent
from agents.extractor import ExtractorAgent
from agents.validator import ValidatorAgent
from agents.corrector import CorrectorAgent
from agents.constructor import ConstructorAgent
from agents.reviewer import ReviewerAgent

# --- Knowledge Base and Utility Imports ---
from knowledge.knowledge_base import (
    deduplicate_data_for_kb,
    update_knowledge_base_file
)

# --- Global Configuration ---
DEEPSEEK_CHAT_CONFIG = "deepseek_chat"
ZHIPU_CHAT_CONFIG = "zhipuai"
KB_JSON_PATH = './knowledge/knowledge_base.json'
RAG_PDF_PATH = './knowledge/bridge.pdf'
ONTOLOGY_TTL_PATH = 'utils1/ontology.ttl'

# --- Logger Setup ---
logger = logging.getLogger("BridgeProcessor")


# --- Helper Classes and Functions for Logging ---

class Color:
    """A class to hold ANSI color codes for colored console output."""
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"


class ColoredFormatter(logging.Formatter):
    """A custom logging formatter to add colors to log levels."""
    LOG_LEVEL_COLORS = {
        logging.DEBUG: Color.CYAN,
        logging.INFO: Color.GREEN,
        logging.WARNING: Color.YELLOW + Color.BOLD,
        logging.ERROR: Color.RED + Color.BOLD,
        logging.CRITICAL: Color.MAGENTA + Color.BOLD,
    }

    def format(self, record):
        color = self.LOG_LEVEL_COLORS.get(record.levelno, Color.RESET)
        record.levelname = f"{color}{record.levelname}{Color.RESET}"
        return super().format(record)


def setup_logging_for_processor(logger_instance):
    """Configures the logger to output to both console and a dated file."""
    # Create a directory for logs based on the current year and month
    current_month_str = datetime.now().strftime("%Y-%m")
    log_dir_path = os.path.join("run", current_month_str)
    os.makedirs(log_dir_path, exist_ok=True)

    # Create a unique log file name with a timestamp
    log_file_name = f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_full_path = os.path.join(log_dir_path, log_file_name)

    logger_instance.setLevel(logging.DEBUG)
    if logger_instance.hasHandlers():
        logger_instance.handlers.clear()

    # Configure console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger_instance.addHandler(console_handler)

    # Configure file handler for detailed debug logging
    file_handler = logging.FileHandler(log_file_full_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - [%(name)s:%(module)s:%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger_instance.addHandler(file_handler)
    logger_instance.info(f"Logging initialized. Log file: {log_file_full_path}")



def append_low_confidence_item(item: Dict[str, Any],
                               file_path: str = os.path.join("run", "low_confidence_manual_review_queue.json")) -> None:
    """Append a low-confidence extraction to the manual-review queue."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    existing_items: List[Dict[str, Any]] = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_items = json.load(f)
            if isinstance(loaded_items, list):
                existing_items = loaded_items
        except (json.JSONDecodeError, OSError):
            existing_items = []

    existing_items.append(item)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_items, f, ensure_ascii=False, indent=4)


def process_single_line(line_content: str, line_idx: int, total_lines_in_topic: int, topic_name: str,
                        agents: Dict[str, Any], kb_json_path: str, kb_lock: Lock) -> List[Dict]:
    """
    Processes a single line of text through the extract-validate-correct pipeline.
    This function is designed to be run in a separate thread.
    """
    line_logger = logging.getLogger("BridgeProcessor.LineWorker")
    line_logger.info(
        f"--- Processing Line in Topic '{topic_name}' ({line_idx}/{total_lines_in_topic}): '{line_content[:100]}{'...' if len(line_content) > 100 else ''}' ---"
    )

    extractor = agents['extractor']
    validator = agents['validator']
    corrector = agents['corrector']

    current_extraction_list = None
    current_extraction_json_str = None

    try:
        # Step 1: Initial Extraction
        # Lock the knowledge base to ensure thread-safe read/load operations.
        with kb_lock:
            extractor.knowledge_base.load_knowledge()
        extracted_items_list = extractor.extract_information(line_content)
        current_extraction_list = extracted_items_list
        current_extraction_json_str = json.dumps(current_extraction_list, ensure_ascii=False)
        line_logger.info(f"Extractor produced {len(extracted_items_list or [])} item(s).")

        # Step 2: Initial Validation
        validation_feedback_json_str, score = validator.validate_and_fuse_extraction(current_extraction_json_str)
        line_logger.info(f"Initial Validation Score: {score}")

        # Step 3: Correction Loop (if necessary)
        # This loop attempts to improve the extraction quality if the initial score is below the threshold.
        max_iterations = 3
        iteration_count = 0
        if score < 1.0:
            line_logger.info("Score < 1.0, entering correction loop...")
            while score < 1.0 and iteration_count < max_iterations:
                iteration_count += 1
                line_logger.info(f"Correction Iteration {iteration_count}...")

                # Correct the extraction based on the validator's feedback
                corrected_extraction_json_str = corrector.correct_extraction(
                    current_extraction_json_str,
                    validation_feedback_json_str
                )
                current_extraction_json_str = corrected_extraction_json_str
                try:
                    current_extraction_list = json.loads(current_extraction_json_str)
                except json.JSONDecodeError as je:
                    line_logger.error(f"Failed to parse corrected JSON in iteration {iteration_count}: {je}")
                    break  # Exit loop if correction produces invalid JSON

                # Re-validate the corrected data to check for improvement
                validation_feedback_json_str, score = validator.validate_and_fuse_extraction(
                    current_extraction_json_str
                )
                line_logger.info(f"Score after Correction Iteration {iteration_count}: {score}")

        # Step 4: Conditionally update the Knowledge Base
        if score >= 1.0:
            # If the data is high-quality, add it to the shared knowledge base.
            line_logger.info(f"Validation successful with score {score}. Updating knowledge base.")
            if current_extraction_list and isinstance(current_extraction_list, list):
                valid_kb_items = [item for item in current_extraction_list if isinstance(item, dict)]
                if valid_kb_items:
                    deduplicated_items = deduplicate_data_for_kb(valid_kb_items)
                    if deduplicated_items:
                        # Use a lock to prevent race conditions when writing to the KB file
                        with kb_lock:
                            line_logger.info(f"Updating KB with {len(deduplicated_items)} verified item(s)...")
                            update_knowledge_base_file(deduplicated_items, kb_json_path)
                            # Reload the KB in this thread's extractor instance to reflect the update
                            extractor.knowledge_base.load_knowledge()
        else:
            try:
                feedback_for_review = json.loads(validation_feedback_json_str)
            except (json.JSONDecodeError, TypeError):
                feedback_for_review = {"raw_feedback": validation_feedback_json_str}
            low_confidence_item = {
                "文本": line_content,
                "topic": topic_name,
                "line_index": line_idx,
                "final_score": score,
                "score_required_for_L_update": 1.0,
                "correction_iterations_used": iteration_count,
                "max_iterations": max_iterations,
                "feedback": feedback_for_review,
                "candidate_extraction": current_extraction_list if isinstance(current_extraction_list, list) else [],
                "action": "manual_review_required; not_added_to_L; not_added_to_final_graph",
            }
            with kb_lock:
                append_low_confidence_item(low_confidence_item)
            line_logger.warning(
                f"Final score {score} (< 1.0). Data will NOT be added to L or the final graph; "
                "it has been routed to the manual-review queue."
            )
            return []

        # Return only fully validated data. Low-confidence data has already been routed to review.
        return current_extraction_list if isinstance(current_extraction_list, list) else []

    except Exception as e:
        line_logger.error(f"Error processing line '{line_content}' in topic '{topic_name}': {e}", exc_info=True)
        return []


def process_report(report_filepath: str, max_workers: int = 4):
    """
    The main function to orchestrate the entire report processing pipeline.
    """
    main_logger = logging.getLogger("BridgeProcessor.Main")
    try:
        with open(report_filepath, 'r', encoding='utf-8') as f:
            report_content_full = f.read()
    except FileNotFoundError:
        main_logger.error(f"Report file not found at {report_filepath}")
        return "Processing Failed: Report file not found."

    # Pre-process the report content
    report_content_full = report_content_full.replace("\\n", "\n")
    report_lines_all = report_content_full.strip().split('\n')
    if not report_lines_all:
        main_logger.error("Report content is empty.")
        return "Processing Failed: Report content is empty."

    # The first line is the bridge name, the rest is content
    bridge_name = report_lines_all[0].strip()
    actual_report_lines_in_original_order = [line.strip() for line in report_lines_all[1:] if line.strip()]

    if not actual_report_lines_in_original_order:
        main_logger.info(f"No actual content lines found for bridge: {bridge_name}. Process finished.")
        return f"Processing complete (no content lines): {bridge_name}"

    actual_report_content_for_decomposer = "\n".join(actual_report_lines_in_original_order)
    main_logger.info(f"Processing report for bridge: {Color.BOLD}{bridge_name}{Color.RESET}")

    # Initialize agents needed for parallel processing
    decomposer = DecomposerAgent(model_config_name=ZHIPU_CHAT_CONFIG)
    agents_for_pool = {
        'extractor': ExtractorAgent(model_config_name=ZHIPU_CHAT_CONFIG, kb_json_path=KB_JSON_PATH, rag_pdf_path=RAG_PDF_PATH),
        'validator': ValidatorAgent(model_config_name=DEEPSEEK_CHAT_CONFIG, kb_json_path=KB_JSON_PATH, ontology_ttl_path=ONTOLOGY_TTL_PATH),
        'corrector': CorrectorAgent(model_config_name=DEEPSEEK_CHAT_CONFIG),
    }

    all_corrected_outputs_for_bridge = []
    kb_access_lock = Lock() # Lock for thread-safe access to the knowledge base file

    # Decompose the report into topics before parallel processing
    main_logger.info("Decomposing report text into topics...")
    decomposed_data_by_topic = decomposer.decompose_text_by_topic(actual_report_content_for_decomposer)
    if not decomposed_data_by_topic or all(not lines for lines in decomposed_data_by_topic.values()):
        main_logger.warning("Decomposer returned no topics. Falling back to a single topic for all lines.")
        decomposed_data_by_topic = {"fallback_topic": actual_report_lines_in_original_order}

    # Use a thread pool to process lines concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_line_info = {}
        # Submit each line from each topic as a separate task
        for topic_name, lines_in_topic in decomposed_data_by_topic.items():
            if not lines_in_topic: continue
            for line_idx, line_content in enumerate(lines_in_topic):
                if not line_content.strip(): continue
                future = executor.submit(process_single_line, line_content.strip(), line_idx + 1, len(lines_in_topic), topic_name, agents_for_pool, KB_JSON_PATH, kb_access_lock)
                future_to_line_info[future] = f"Topic: {topic_name}, Line: {line_content.strip()[:50]}..."

        # Collect results as they are completed
        for future in concurrent.futures.as_completed(future_to_line_info):
            line_info = future_to_line_info[future]
            try:
                corrected_outputs_for_line = future.result()
                if corrected_outputs_for_line:
                    all_corrected_outputs_for_bridge.extend(corrected_outputs_for_line)
                main_logger.info(f"Successfully processed task for: {line_info}")
            except Exception as exc:
                main_logger.error(f"Task for {line_info} generated an exception: {exc}", exc_info=True)

    main_logger.info(f"--- Parallel processing complete. Collected {len(all_corrected_outputs_for_bridge)} items. ---")

    # --- Sequential post-processing steps ---
    reviewer = ReviewerAgent(model_config_name=DEEPSEEK_CHAT_CONFIG)
    constructor = ConstructorAgent(model_config_name=DEEPSEEK_CHAT_CONFIG)

    # Perform a final review on the aggregated, corrected data
    main_logger.info("Reviewer is checking all corrected data for final consistency...")
    corrected_data_json_str = json.dumps(all_corrected_outputs_for_bridge, ensure_ascii=False)
    reviewed_output_json_str = reviewer.review_constructed_data(corrected_data_json_str)
    try:
        final_data_list = json.loads(reviewed_output_json_str)
        if not isinstance(final_data_list, list): final_data_list = [final_data_list] if final_data_list else []
    except json.JSONDecodeError as e:
        main_logger.error(f"Failed to parse FINAL reviewed output JSON: {e}")
        final_data_list = []

    # Sort the final output to match the original report's line order
    main_logger.info("Sorting final outputs according to original report order...")
    output_items_by_original_text = {}
    for item in final_data_list:
        original_text_key = item.get("文本", "").strip()
        if original_text_key:
            if original_text_key not in output_items_by_original_text:
                output_items_by_original_text[original_text_key] = []
            output_items_by_original_text[original_text_key].append(item)

    sorted_outputs = []
    for original_line_text in actual_report_lines_in_original_order:
        if original_line_text in output_items_by_original_text:
            sorted_outputs.extend(output_items_by_original_text[original_line_text])

    all_final_outputs_for_bridge_sorted = sorted_outputs
    main_logger.info(f"--- Finished processing all topics for bridge: {Color.BOLD}{bridge_name}{Color.RESET} ---")

    # Save the final, sorted, and reviewed data to a JSON file
    final_output_filename = os.path.join("./data", bridge_name, f"{bridge_name}_final_reviewed_output.json")
    try:
        os.makedirs(os.path.dirname(final_output_filename), exist_ok=True)
        with open(final_output_filename, 'w', encoding='utf-8') as f:
            json.dump(all_final_outputs_for_bridge_sorted, f, ensure_ascii=False, indent=4)
        main_logger.info(f"Sorted final reviewed output saved to: {final_output_filename}")
    except Exception as e:
        main_logger.error(f"Failed to save final output to {final_output_filename}: {e}", exc_info=True)

    # Construct the final knowledge graph in Neo4j
    try:
        main_logger.info("Clearing existing graph data in Neo4j...")
        constructor.clear_database()
        main_logger.info(f"Constructing graph from {len(all_final_outputs_for_bridge_sorted)} final items...")
        constructor.construct_graph(all_final_outputs_for_bridge_sorted, bridge_name)
        main_logger.info(f"Graph construction complete for bridge '{bridge_name}'.")
    except Exception as e:
        main_logger.error(f"An error occurred during graph construction: {e}", exc_info=True)
    finally:
        constructor.close()

    return f"Processing complete for bridge: {bridge_name}. Outputs saved in ./data/{bridge_name}/"


if __name__ == "__main__":
    setup_logging_for_processor(logging.getLogger("BridgeProcessor"))
    report_file = './data/inspection_report.txt'

    start_time = time.time()
    result_message = process_report(report_file, max_workers=2)
    end_time = time.time()

    logging.getLogger("BridgeProcessor").info(result_message)
    logging.getLogger("BridgeProcessor").info(f"Total processing time: {end_time - start_time:.2f} seconds")
