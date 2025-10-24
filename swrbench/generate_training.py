import concurrent.futures
import threading
from tqdm import tqdm
import os
import json
import logging

from utils import run_chat, safe_parse_time, retry_function, save_jsonl, load_jsonl, save_json, load_json
import pr_agent


log_file = "generate_training.log"
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s',  # Added filename and lineno
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

# Initialize logger for pr_agent module after basicConfig
pr_agent.LOGGER = logging.getLogger('pr_agent')
pr_agent.MAX_TOKENS = 16000

CHAT_MODEL = 'gemini-2.5-flash-preview-04-17'
CHAT_TEMPERATURE = 0.7

CHANGE_PROMPT = """
You are an AI assistant tasked with generating training data based on provided Code Review metadata. This metadata includes a pull request's code diff and a list of "Changes" (ground-truth review comments identified by human reviewers).

The input will be provided as follows:

<START_OF_CODE_DIFF>
{code_diff_data}
<END_OF_CODE_DIFF>

<START_OF_PR_CHANGES>
{pr_changes_data}
<END_OF_PR_CHANGES>

Your task is to generate training data that will be used to train a code review agent. This agent will learn to perform code reviews on pull requests and produce a code review report that incorporates the ground-truth PR_CHANGES.

You need to provide two distinct sections in your output:

1.  **`Analysis Process:`**
    *   In this section, analyze the code diff provided within `<START_OF_CODE_DIFF>`.
    *   Adopt the persona of an expert code reviewer who is seeing this code diff for the first time.
    *   It should start with "Okay, let's review this " or "Okay, let's examine this ", etc.
    *   Provide a step-by-step explanation detailing the thought process you (as the reviewer) would follow when examining **only** the code diff. This reasoning should lead you to identify areas for improvement or specific changes.
    *   The insights and suggestions derived from your simulated review process should ultimately align with the "Changes" (improvements) listed in `<START_OF_PR_CHANGES>`.
    *   **Crucially, your entire explanation within this `Analysis Process` section must be framed as if you are performing a blind code review. You must NOT state, imply, or hint that you have any prior knowledge of the ground-truth changes from `<START_OF_PR_CHANGES>` or that you know what the final, correct changes are supposed to be. Your narrative should reflect a reviewer discovering these points for the first time, based solely on the evidence within the code diff.**
    *   Focus on how a reviewer would deduce these potential changes by observing patterns, potential issues, or opportunities for improvement directly from the code diff.
    *   This section is about demonstrating the *deductive reasoning* of a reviewer performing a standard review, not about explaining the mechanics of training data generation or acknowledging the ground-truth source.

2.  **`YAML Code Review Report:`**
    *   In this section, generate a YAML-formatted code review report.
    *   This report MUST accurately reflect the improvements specified in `<START_OF_PR_CHANGES>`.
    *   The report must strictly adhere to the Pydantic schema defined below for `$PRReview`.

The format for the `YAML Code Review Report` is as follows:
The output for this section must be a YAML object equivalent to type $PRReview, according to the following Pydantic definitions:
=====

class KeyIssuesComponentLink(BaseModel):
    relevant_file: str = Field(description="The full file path of the relevant file")
    issue_header: str = Field(description="One or two word title for the issue. For example: 'Possible Bug', etc.")
    issue_content: str = Field(description="A short and concise summary of what should be further inspected and validated during the PR review process for this issue. Do not reference line numbers in this field.")
    start_line: int = Field(description="The start line that corresponds to this issue in the relevant file")
    end_line: int = Field(description="The end line that corresponds to this issue in the relevant file")

class Review(BaseModel):
    estimated_effort_to_review_[1-5]: int = Field(description="Estimate, on a scale of 1-5 (inclusive), the time and effort required to review this PR by an experienced and knowledgeable developer. 1 means short and easy review , 5 means long and hard review. Take into account the size, complexity, quality, and the needed changes of the PR code diff.")
    relevant_tests: str = Field(description="yes\\no question: does this PR have relevant tests added or updated ?")
    key_issues_to_review: List[KeyIssuesComponentLink] = Field("A short and diverse list (0-3 issues) of high-priority bugs, problems or performance concerns introduced in the PR code, which the PR reviewer should further focus on and validate during the review process.")
    security_concerns: str = Field(description="Does this PR code introduce possible vulnerabilities such as exposure of sensitive information (e.g., API keys, secrets, passwords), or security concerns like SQL injection, XSS, CSRF, and others ? Answer 'No' (without explaining why) if there are no possible issues. If there are security concerns or issues, start your answer with a short header, such as: 'Sensitive information exposure: ...', 'SQL injection: ...' etc. Explain your answer. Be specific and give examples if possible")

class PRReview(BaseModel):
    review: Review
=====

Example output for the `YAML Code Review Report` section:
```yaml
review:
  estimated_effort_to_review_[1-5]: |
    3
  relevant_tests: |
    No
  key_issues_to_review:
    - relevant_file: |
        directory/xxx.py
      issue_header: |
        Possible Bug
      issue_content: |
        ...
      start_line: 12
      end_line: 14
    - ...
  security_concerns: |
    No
```

The `YAML Code Review Report` section should contain only the valid YAML, and nothing else. Each YAML output MUST start on a new line, be properly indented, and use block scalar indicators ('|') for multi-line strings where appropriate, as shown in the example.
"""

CLEAN_PROMPT = """
You are an AI assistant tasked with generating training data based on provided Code Review metadata. This metadata includes a pull request's code diff. The PR is considered "clean", meaning it is presumed to have no significant issues or defects.

The input will be provided as follows:

<START_OF_CODE_DIFF>
{code_diff_data}
<END_OF_CODE_DIFF>

Your task is to generate training data that will be used to train a code review agent. This agent will learn to perform code reviews on pull requests.

You need to provide two distinct sections in your output:

1.  **`Analysis Process:`**
    *   In this section, analyze the code diff provided within `<START_OF_CODE_DIFF>`.
    *   Provide a step-by-step explanation of why the code is considered "clean". For example, you might comment on adherence to coding standards, clarity, the presence of tests, or the lack of obvious bugs.
    *   This should clearly illustrate the reasoning process for deeming the PR clean. You should act as a code review expert, demonstrating your reasoning for deeming the PR clean as if you are performing the code review. This is not about explaining how to generate training data.

2.  **`YAML Code Review Report:`**
    *   In this section, generate a YAML-formatted code review report.
    *   Since the PR is "clean", the `key_issues_to_review` section should ideally be empty or explicitly state that no major issues were found.
    *   The `security_concerns` should reflect that no vulnerabilities were identified.
    *   The report must strictly adhere to the Pydantic schema defined below for `$PRReview`.

The format for the `YAML Code Review Report` is as follows:
The output for this section must be a YAML object equivalent to type $PRReview, according to the following Pydantic definitions:
=====

class KeyIssuesComponentLink(BaseModel):
    relevant_file: str = Field(description="The full file path of the relevant file")
    issue_header: str = Field(description="One or two word title for the issue. For example: 'Possible Bug', etc.")
    issue_content: str = Field(description="A short and concise summary of what should be further inspected and validated during the PR review process for this issue. Do not reference line numbers in this field.")
    start_line: int = Field(description="The start line that corresponds to this issue in the relevant file")
    end_line: int = Field(description="The end line that corresponds to this issue in the relevant file")

class Review(BaseModel):
    estimated_effort_to_review_[1-5]: int = Field(description="Estimate, on a scale of 1-5 (inclusive), the time and effort required to review this PR by an experienced and knowledgeable developer. 1 means short and easy review , 5 means long and hard review. Take into account the size, complexity, quality, and the needed changes of the PR code diff.")
    relevant_tests: str = Field(description="yes\\no question: does this PR have relevant tests added or updated ?")
    key_issues_to_review: List[KeyIssuesComponentLink] = Field("A short and diverse list (0-3 issues) of high-priority bugs, problems or performance concerns introduced in the PR code, which the PR reviewer should further focus on and validate during the review process. For a clean PR, this list should be empty.")
    security_concerns: str = Field(description="Does this PR code introduce possible vulnerabilities such as exposure of sensitive information (e.g., API keys, secrets, passwords), or security concerns like SQL injection, XSS, CSRF, and others ? Answer 'No' (without explaining why) if there are no possible issues. If there are security concerns or issues, start your answer with a short header, such as: 'Sensitive information exposure: ...', 'SQL injection: ...' etc. Explain your answer. Be specific and give examples if possible")

class PRReview(BaseModel):
    review: Review
=====

Example output for the `YAML Code Review Report` section for a clean PR:
```yaml
review:
  estimated_effort_to_review_[1-5]: |
    1
  relevant_tests: |
    Yes
  key_issues_to_review: []
  security_concerns: |
    No
```

The `YAML Code Review Report` section should contain only the valid YAML, and nothing else. Each YAML output MUST start on a new line, be properly indented, and use block scalar indicators ('|') for multi-line strings where appropriate, as shown in the example.
"""

def process_item(item_details_tuple):
    item, data_type, results_path, lock, chat_model_to_use, change_prompt_template, clean_prompt_template, chat_temp = item_details_tuple
    
    item_id = item.get('instance_id') # Assumes instance_id is pre-populated

    try:
        repo_path = f"/SWRBench/data_train/projects/{item['repo'].replace('/', '__')}/{item_id}"
        if not os.path.exists(repo_path):
            logging.warning(f"Repository not found: {repo_path} for item {item_id}")
            return

        base_branch_name = 'base_branch'
        target_branch_name = 'branch_under_review'

        # As per original logic, TokenHandler is initialized with CHANGE_PROMPT template
        token_handler = pr_agent.TokenHandler(chat_model_to_use, change_prompt_template, change_prompt_template)
        
        diff_content, diff_files = pr_agent.get_pr_diff(repo_path, base_branch_name, target_branch_name, 
                                token_handler, chat_model_to_use, add_line_numbers_to_hunks=True)
        
        if not diff_content:
            logging.info(f"No code changes or significant commit messages detected, nothing to review for {item_id}.")
            return
        
        current_prompt_str = ""
        if data_type == "change":
            pr_changes_data = []
            for defect_item in item['defects']:
                change_data_obj = {
                    "change_introduction": {
                        'code_snippet': defect_item['change_introduction']['code_snippet'],
                    },
                    "discussion_evidence": {
                        'discussion_summary': defect_item['discussion_evidence']['discussion_summary'],
                        'original_reviewer_comment': defect_item['discussion_evidence']['original_reviewer_comment'],
                    },
                }
                if "resolving_information" in defect_item:
                    change_data_obj["resolving_information"] = {
                        "code_snippet": defect_item['resolving_information']['code_snippet'], 
                        "resolution_explanation": defect_item['resolving_information']['resolution_explanation']
                    }
                pr_changes_data.append(change_data_obj)
                
            current_prompt_str = change_prompt_template.format(code_diff_data=diff_content, pr_changes_data=json.dumps(pr_changes_data, indent=2, ensure_ascii=False))
        elif data_type == "clean":
            current_prompt_str = clean_prompt_template.format(code_diff_data=diff_content)
        else:
            logging.error(f"Invalid data type: {data_type}")
            raise ValueError(f"Invalid data type: {data_type}")
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "defects", # Generic name for the expected output structure
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "analysis_process": {"type": "string"},
                        "yaml_code_review_report": {"type": "string"},
                    },
                    "required": ["analysis_process", "yaml_code_review_report"],
                    "additionalProperties": False
                }
            }
        }
        
        json_str = run_chat(
            chat_model_to_use, 
            [{"role": "user", "content": current_prompt_str}], 
            temperature=chat_temp, 
            response_format=response_format,
            max_retries=3
        )
        result = json.loads(json_str)
            
        assert result['yaml_code_review_report'] is not None
        assert result['analysis_process'] is not None
        
        analysis_process = result['analysis_process']
        
        if "ground-truth" in analysis_process.lower() or "ground truth" in analysis_process.lower():
            logging.error(f"Ground-truth found in analysis_process for {item_id}")
            return
        
        yaml_review = result['yaml_code_review_report']
        review = pr_agent.parse_review(yaml_review, diff_files)
        training_data = {
            "instance_id": item_id,
            "title": item['pr_title'],
            "type": data_type,
            "analysis_process": analysis_process,
            "code_review_yaml": yaml_review,
            "code_review": review,
            "code_diff": diff_content,
        }
        with lock:
            with open(results_path, "a") as f:
                f.write(json.dumps(training_data) + "\n")
            
    except Exception as e:
        current_item_id_for_log = item_id if item_id else item.get('repo', 'unknown_repo') + '-' + str(item.get('pr_number', 'unknown_pr'))
        logging.error(f"Error processing item {current_item_id_for_log}: {e}", exc_info=True)


def generate_training_data(input_data_path, results_path, data_type, max_workers=None):

    input_data = load_jsonl(input_data_path)

    # Pre-populate instance_id for all items
    for item_entry in input_data:
        item_entry["instance_id"] = f"{item_entry['repo'].replace('/', '__')}-{item_entry['pr_number']}"
    
    prior_results_ids = set()
    if os.path.exists(results_path):
        try:
            prior_results = load_jsonl(results_path)
            logging.info(f"Prior results: {len(prior_results)}")
            prior_results_ids = {e['instance_id'] for e in prior_results if 'instance_id' in e}
        except Exception as ex_load:
            logging.error(f"Error loading prior results from {results_path}: {ex_load}. Assuming no prior results.", exc_info=True)
            # Fallthrough to use all input_data if prior results are unloadable
    
    remain_data = [e for e in input_data if e['instance_id'] not in prior_results_ids]
    
    if not remain_data:
        logging.info("No new data to process.")
        return
    logging.info(f"Remaining data to process: {len(remain_data)}")
    
    if data_type not in ["change", "clean"]:
        logging.error(f"Invalid data type: {data_type}")
        raise ValueError(f"Invalid data type: {data_type}")
    
    file_lock = threading.Lock()

    if max_workers is None:
        cpu_count = os.cpu_count()
        max_workers = min(32, cpu_count + 4 if cpu_count else 8) # Default max_workers
    logging.info(f"Using {max_workers} worker threads for {data_type} data generation.")

    tasks_to_submit = []
    for item_data in remain_data:
        # item_data already has 'instance_id'
        tasks_to_submit.append((item_data, data_type, results_path, file_lock, 
                               CHAT_MODEL, CHANGE_PROMPT, CLEAN_PROMPT, CHAT_TEMPERATURE))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item_id = {
            executor.submit(process_item, task_args): task_args[0]['instance_id'] 
            for task_args in tasks_to_submit
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_item_id), total=len(tasks_to_submit), desc=f"Generating training data ({data_type})"):
            item_id_completed = future_to_item_id[future]
            try:
                future.result() # Retrieve result, re-raises exceptions from process_item if not caught there.
                                # Since process_item logs its own errors, this is mainly to ensure completion.
            except Exception as exc_future:
                # This catches errors if process_item itself crashed before its own try-except or re-raised.
                logging.error(f"Item {item_id_completed} processing generated an unexpected exception in executor: {exc_future}", exc_info=True)


if __name__ == "__main__":
    # Example: Run with a specific number of workers
    # num_workers = 4 
    num_workers = 16 # Use default calculation

    results_path_change = "/SWRBench/data_train/training_change_results.jsonl"
    change_data_path = "/SWRBench/data_train/verified_defect_results_all_sampled_0521.jsonl"
    logging.info(f"Starting 'change' data generation. Results will be saved to {results_path_change}")
    generate_training_data(change_data_path, results_path_change, "change", max_workers=num_workers)
    logging.info(f"Finished 'change' data generation.")
    
    results_path_clean = "/SWRBench/data_train/training_clean_results.jsonl"
    clean_data_path = "/SWRBench/data_train/verified_clean_results_all_sampled_0521.jsonl"
    logging.info(f"Starting 'clean' data generation. Results will be saved to {results_path_clean}")
    generate_training_data(clean_data_path, results_path_clean, "clean", max_workers=num_workers)
    logging.info(f"Finished 'clean' data generation.")
