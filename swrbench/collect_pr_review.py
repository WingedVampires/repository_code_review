import os
import json
import random
import re
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from utils import run_chat, safe_parse_time, retry_function, save_jsonl, load_jsonl, save_json, load_json


CHAT_MODEL = 'gemini-2.5-flash-preview-04-17'
# PARSER_MODEL = 'gemini-2.0-flash-exp'
# PARSER_MODEL = 'deepseek-chat'
# PARSER_MODEL = 'gpt-4o-mini'
PARSER_MODEL = 'gemini-2.5-flash-preview-04-17'
OPENAI_TEMPERATURE = 0.7

def get_fix_commits(repo_name):
    pr_jsonl_file = f'data/prs_raw/{repo_name.replace("/", "__")}__prs.jsonl'
    if not os.path.exists(pr_jsonl_file):
        print(f"{repo_name} has no PR information")
        return
    prs = load_jsonl(pr_jsonl_file)
    prs = [pr for pr in prs if not pr.get('ERROR', False)]
    fix_prs = []
    for pr in prs:
        pr_title = pr['title']
        if ' fix ' in pr_title.lower():
            fix_prs.append(pr)
    print(f"{repo_name} has {len(fix_prs)} fix PRs")
    
    fix_commits = []
    for pr in fix_prs:
        commits = [item for item in pr['timeline'] if item['type'] == 'commit']
        for commit in commits:
            created_date = safe_parse_time(pr['created_at'])
            earliest_issue_date = (created_date - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S")
            
            fix_commits.append({
                'fix_commit_hash': commit['sha'],
                'repo_name': repo_name.replace("/", "__"),
                "earliest_issue_date": earliest_issue_date
            })
            
    print(f"{repo_name} 共有 {len(fix_commits)} 个修复commit")
    with open(f'data/{repo_name.replace("/", "__")}/fix_commits.json', 'w') as f:
        json.dump(fix_commits, f, indent=4)


def filter_prs(repo_name, prs, max_commits=100, pr_ids_range=None):
    # target_results_paath = '/SWRBench/data/verified_results_label_0412.jsonl'
    # target_results = load_jsonl(target_results_paath)
    # target_pr_ids = [repo_name + '_' + str(pr['pr_number']) for pr in target_results]
    # target_pr_ids = [
    #     'astropy/astropy_2372', 'astropy/astropy_2427', 'astropy/astropy_2534', 'astropy/astropy_2626', 
    #     'astropy/astropy_2785', 'astropy/astropy_2846', 'astropy/astropy_3185', 'astropy/astropy_3396', 
    #     'astropy/astropy_3767', 'astropy/astropy_3789', 'astropy/astropy_3993', 'astropy/astropy_4312', 
    #     'astropy/astropy_4784', 'astropy/astropy_5106', 'astropy/astropy_5521', 'astropy/astropy_5760', 
    #     'astropy/astropy_5996', 'astropy/astropy_6522', 'astropy/astropy_7267', 'astropy/astropy_7463', 
    #     'astropy/astropy_7538', 'astropy/astropy_8441', 'astropy/astropy_8564', 'astropy/astropy_8838', 
    #     'astropy/astropy_9354', 'astropy/astropy_9577', 'astropy/astropy_10005', 'astropy/astropy_10126', 
    #     'astropy/astropy_11664', 'astropy/astropy_11860', 'astropy/astropy_11943', 
    # ]
    
    filtered_prs = []
    filter_count = {}
    for pr in prs:
        if pr.get('ERROR', False):  # 跳过失败的PR
            filter_count['ERROR'] = filter_count.get('ERROR', 0) + 1
            continue
        #没有合并
        # if not pr.is_merged:
        #     continue
        #不在指定的pr 的 id范围内
        if pr_ids_range is not None and (pr['number'] < pr_ids_range[0] or pr['number'] > pr_ids_range[1]):
            filter_count['pr_ids_range'] = filter_count.get('pr_ids_range', 0) + 1
            continue
        # 获取PR信息
        comments = [c for c in pr['timeline'] if 'comment' in c['type']]
        if len(comments) < 2:
            filter_count['comments'] = filter_count.get('comments', 0) + 1
            continue
        if 'head_ref_force_pushed' in pr['issue_events']:
            filter_count['head_ref_force_pushed'] = filter_count.get('head_ref_force_pushed', 0) + 1
            continue
        
        pr_description = [e for e in pr['timeline'] if e['type'] == 'description']
        if len(pr_description) == 0:
            filter_count['pr_description_filter'] = filter_count.get('pr_description_filter', 0) + 1
            continue
        
        empty_commits = [e for e in pr['timeline'] if e['type'] == 'commit' and len(e['diff']) == 0]
        if len(empty_commits) > 0:
            filter_count['empty_commit_filter'] = filter_count.get('empty_commit_filter', 0) + 1
            continue
        
        #不在指定的commit的范围内
        if pr['commits'] <= 1:
            filter_count['commits_1'] = filter_count.get('commits_1', 0) + 1
            continue
        if pr['commits'] > max_commits:
            filter_count['commits_gt_max'] = filter_count.get('commits_gt_max', 0) + 1
            continue
        # if repo_name + '_' + str(pr['number']) not in target_pr_ids:
        #     filter_count['target_pr_ids'] = filter_count.get('target_pr_ids', 0) + 1
        #     continue
        
        filtered_prs.append(pr)
    
    print(f"过滤统计: {filter_count}")
    return filtered_prs



def construct_merged_prompt(pr_info):
    # Extract PR description
    pr_description = next((item['body'] for item in pr_info.get('timeline') if item.get('type') == 'description'), "无描述")

    # Format the timeline information chronologically
    timeline_info = ""
    # Sort timeline items by date/time if not already sorted
    # Assuming 'created_at' or 'date' exists and is comparable
    def get_timestamp(item):
        return item.get('created_at') or item.get('date')

    sorted_timeline = sorted(
        [item for item in pr_info.get('timeline', []) if get_timestamp(item)], 
        key=get_timestamp
    )

    for item in sorted_timeline:
        item_type = item.get('type')
        if item_type == 'description':
            continue # Already captured
        elif item_type == 'comment':
            timeline_info += f"<Start of Comment>\nTime: {item.get('created_at')}\nAuthor: {item.get('user')}\nComment: {item.get('body')}\n<End of Comment>\n\n"
        elif item_type == 'review_comment':
            reply_comments = ""
            # Sort replies chronologically if necessary (assuming 'reply' is a list of dicts with 'created_at')
            sorted_replies = sorted(item.get('reply'), key=lambda c: c.get('created_at'))
            for comment in sorted_replies:
                 reply_comments += f"<Start of Sub Review Comment>\nTime: {comment.get('created_at')}\nAuthor: {comment.get('user')}\nComment: {comment.get('body')}\n<End of Sub Review Comment>\n"
            # Add the main review comment itself, which triggered the replies
            review_comment_text = f"Time: {item.get('created_at')}\nAuthor: {item.get('user')}\nComment: {item.get('body')}\n" # Added the main comment
            timeline_info += f"<Start of Review Comment>\n{review_comment_text}<Start of Related Diff Hunk>\nFile: {item.get('path')}\n{item.get('diff_hunk')}\n<End of Related Diff Hunk>\n{reply_comments}<End of Review Comment>\n\n"
        elif item_type == 'commit':
            timeline_info += f"<Start of Commit>\nTime: {item.get('date')}\nSHA: {item.get('sha')}\nAuthor: {item.get('author')}\nMessage: {item.get('message')}\nDiff:\n{item.get('diff_text')}\n<End of Commit>\n\n"
        elif item_type == 'review':
             timeline_info += f"<Start of Review>\nTime: {item.get('created_at')}\nAuthor: {item.get('user')}\nState: {item.get('state')}\nReview Body: {item.get('body')}\n<End of Review>\n\n"
            
    # Map defect types for clarity in the prompt
    defect_type_text_map = {
        'E.1.1': 'E.1.1 Textual Changes', 'E.1.2': 'E.1.2 Language Features',
        'E.2': 'E.2 Visual Representation',
        'E.3.1': 'E.3.1 Organization', 'E.3.2': 'E.3.2 Solution Approach',
        'F.1': 'F.1 Interface', 'F.2': 'F.2 Logic', 'F.3': 'F.3 Resource',
        'F.4': 'F.4 Check', 'F.5': 'F.5 Support', 'F.6': 'F.6 Larger Defects',
    }
    defect_categories_description = """
- **E. Evolvability Changes**: Modifications improving code readability, maintainability, or overall structure without altering core functionality.
  - **E.1 Documentation**: Enhancements to code comprehension for developers through documentation elements.
    - *E.1.1 Textual Changes*: Adjustments to comments (e.g., adding, correcting, clarifying) or identifier names (variables, functions, classes) for better clarity and consistency.
    - *E.1.2 Language Features*: Utilizing language-specific constructs (e.g., `final` in Java, type annotations, access modifiers) primarily to convey developer intent, constraints, or information, rather than for functional impact.
  - **E.2 Visual Representation**: Modifications to code formatting and layout, such as indentation, spacing, line breaks, or bracket placement, to improve visual clarity and adhere to style conventions.
  - **E.3 Structure**: Changes affecting the organization or implementation strategy of the code without changing its external behavior.
    - *E.3.1 Organization*: Reorganizing code elements, such as removing dead (unused) code, moving functions or classes to more appropriate locations, or restructuring files/packages for better modularity.
    - *E.3.2 Solution Approach*: Modifying the internal implementation details or algorithms (e.g., refactoring for clarity/efficiency, updating function usage to newer patterns), or adding supporting code like tests, without altering the observable functionality.

- **F. Functional Changes**: Corrections addressing errors in the code's behavior, output, or interaction with other system parts.
  - **F.1 Interface**: Fixes related to how different code components interact, including incorrect method calls, wrong parameter types/values, violated API contracts, or incorrect event handling.
  - **F.2 Logic**: Corrections to errors in algorithms, conditional statements (if/else), loops, computations, or other logical constructs leading to incorrect behavior.
  - **F.3 Resource**: Fixes concerning the management of data, variables, or system resources, including initialization errors, memory leaks, improper resource release/acquisition, or incorrect data manipulation (e.g., concurrency issues).
  - **F.4 Check**: Adding or modifying validation or checks (e.g., null checks, boundary checks, state validation) for variables, parameters, or function return values to handle potential errors or invalid states correctly.
  - **F.5 Support**: Corrections related to the interaction with external systems, libraries, frameworks, or APIs (e.g., incorrect usage, adapting to API changes, version incompatibilities).
  - **F.6 Larger Defects**: Significant functional fixes that often span multiple files or components, address incompletely implemented features, fix major inconsistencies (like GUI behavior), or require broader system knowledge. These might not always be fully resolved within the same PR.
"""

    prompt = f"""
You are an expert code review analyst. Your task is to analyze the provided Pull Request (PR) information, focusing on the chronological sequence of commits, comments, and reviews. Your goal is to identify **ALL** instances where a **reviewer** either:
a) Discovered a **Defect** (an error or flaw) that was introduced within this PR and subsequently fixed by the developer.
b) **Suggested a Change** (an improvement for readability, maintainability, performance, style, alternative approach, etc., *not necessarily an error*) related to code introduced in this PR, which was subsequently implemented by the developer.

Both Defects and Suggested Changes must be addressed (fixed or implemented) in a later commit within the same PR (with a note for F.6 defects regarding the fix timeline).

**Categories for Changes and Defects:**
Carefully review the following change and defect categories to classify any findings:
{defect_categories_description}

**Conditions for a Change/Defect to Qualify:**
For a change/defect to be identified and reported, **ALL** of the following conditions must be strictly met (with a note for F.6 regarding the fix timeline):
1.  **Introduction:** The change/defect must be demonstrably introduced by an early commit *within this specific PR*. It should not be pre-existing code. Verify this using the commit diffs provided in the timeline.
2.  **Reviewer Discovery/Suggestion:** A **reviewer** (not the PR author) must explicitly point out the change/defect in a review comment or general PR comment. There must be clear textual evidence of the reviewer identifying the issue or suggesting the change.
3.  **Discussion/Confirmation:** The discussion should confirm that a change/defect was indeed found. Ideally, the developer acknowledges the issue or the fix addresses the reviewer's specific point.
4.  **Fix/Implementation:** One or more subsequent commits *within the same PR* must contain changes that demonstrably fix the specific change/defect identified by the reviewer. (Exception: F.6 Larger Defects might not be fully fixed within the PR, but the discussion must still identify them as significant issues found by the reviewer).
5.  **Timeline:** The introduction, reviewer discussion, and fix or implementation (except potentially for F.6) must all occur within the lifecycle of this PR.

**Analysis Steps:**
1.  Carefully read the entire PR timeline (commits, comments, reviews) in chronological order.
2.  Identify potential defects or areas for suggested changes introduced in early commits.
3.  Look for **explicit reviewer comments** that point out these specific changes/defects. **Do NOT miss any changes/defects pointed out by reviewers.** Pay close attention to review comments associated with diff hunks.
4.  Verify if the identified issue meets **ALL** the qualification criteria (Introduction, Reviewer Discovery, Discussion, Fix/Implementation within PR).
5.  For each qualifying change/defect found, classify it using the most specific fine-grained category (e.g., E.1.1, F.2).
6.  Extract the required information for the output format below.

**PR Information:**
- **Number**: {pr_info.get('number')}
- **Title**: {pr_info.get('title')}
- **Author**: {pr_info.get('user')}
- **Created**: {pr_info.get('created_at')}
- **Merged**: {pr_info.get('merged_at')}

**PR Description:**
{pr_description}

**PR Timeline (Processed Chronologically):**
{timeline_info}

**Output Instructions:**

**If NO changes/defects meeting ALL the criteria are found:**
Simply respond with empty list:
"[]"

**If one or more changes/defects meeting ALL the criteria ARE found:**
   - Generate a JSON list. Each object in the list represents one qualifying change/defect.
   - **Structure for EACH change/defect object in the JSON list:**
     ```json
     [
        {{
        "change_type": "string - The full, fine-grained change type classification (e.g., 'E.1.1 Textual Changes', 'F.2 Logic', 'F.3 Resource', 'F.6 Larger Defects').",
        "change_introduction": {{
            "commit_sha": "string - The exact SHA of the commit that introduced the change/defect.",
            "code_snippet": "string - The relevant lines of code (from the diff) that introduced the change/defect. Format as a multi-line string if necessary."
        }},
        "discussion_evidence": {{
            "first_mention_timestamp": "string - The timestamp (e.g., from 'created_at') when the reviewer first mentioned this specific change/defect.",
            "original_reviewer_comment": "string - The exact and complete comment text from the reviewer who pointed out this specific change/defect. Include context if necessary (e.g., associated diff hunk for review comments).",
            "discussion_summary": "string - A concise summary of how the change/defect was discussed, including developer acknowledgment or confirmation of it being a valid issue/change."
        }},
        "severity_assessment": {{
            "justification": "string - Your reasoning for the chosen severity score (1-10). Base this on potential impact (crashes, incorrect results, maintainability, user impact), scope (core vs. minor feature), and discussion tone/urgency.",
            "severity_level": "integer - Rate severity from 1 to 10. Provide only the number. Scale: 1-2 (Trivial, e.g., minor typo in comment), 3-4 (Minor, e.g., slightly inefficient code, unclear logging), 5-6 (Moderate, e.g., specific edge case error, noticeable performance issue), 7-8 (Major, e.g., incorrect logic, crash in non-core function), 9-10 (Critical, e.g., core crash, security vulnerability)."
        }},
        "resolving_information": {{ // IMPORTANT: This entire object is CONDITIONAL.
                                    // OMIT this "resolving_information" field ENTIRELY if "change_type" is "F.6 Larger Defects".
                                    // For all other types, INCLUDE this field if the information is present.
            "commit_sha": "string - The exact SHA of the commit that fixed the defect or implemented the change.",
            "code_snippet": "string - The relevant lines of code (from the diff) that fixed the defect or implemented the change. Format as a multi-line string.",
            "resolution_explanation": "string - A precise explanation of how the code change in the fixing commit addresses the specific change/defect identified by the reviewer."
        }}
        }},
        // ... more change/defect objects if present
    ]
     ```

**Core Criteria for a Qualifying Change/Defect:**
A change/defect MUST meet ALL the following conditions to be included in the JSON output:
*   **Reviewer Discovery:** Initially identified and pointed out by a reviewer (NOT by the author themselves).
*   **Introduction Evidence:** Clear evidence of how and when it was introduced (commit SHA, code snippet).
*   **Discussion Confirmation:** Evidence of discussion confirming it as a valid issue/change (reviewer comment, summary).
*   **Resolution (with F.6 exception):** Generally, it must be resolved within the scope of the analyzed context (e.g., within the Pull Request). The "F.6 Larger Defects" type is an exception and does not require `resolving_information`.
**Critical Instructions & Reminders:**
*   **ACCURACY IS PARAMOUNT:** Ensure all extracted data (SHAs, comments, code snippets, timestamps) is exact and complete.
*   **FOCUS EXCLUSIVELY ON REVIEWER-DISCOVERED ITEMS:** Do NOT include issues found by the author, self-corrected issues before review, or suggestions not related to a defect/change.
*   **STRICT CRITERIA ADHERENCE:** If a potential item does not meet ALL criteria (considering the F.6 exception for resolution), do not include it.
*   **COMPLETENESS:** Analyze the *entire* provided text and report ALL qualifying changes/defects. Do not stop after finding the first one.
*   **AVOID DUPLICATION / ENSURE ATOMICITY:** If multiple comments or discussion points clearly refer to the *exact same underlying change or defect*, consolidate them into a single, comprehensive entry. Do not split one conceptual change/defect into multiple reported items. Identify the core issue.
*   **JSON VALIDITY:** Ensure the final output is a valid JSON list.

"""

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "defects",
            "strict": True,
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "change_type": {"type": "string"},
                        "change_introduction": {
                            "type": "object",
                            "properties": {
                                "commit_sha": {"type": "string"},
                                "code_snippet": {"type": "string"},
                            },
                            "required": ["commit_sha", "code_snippet"],
                            "additionalProperties": False
                        },
                        "discussion_evidence": {
                            "type": "object",
                            "properties": {
                                "first_mention_timestamp": {"type": "string"},
                                "original_reviewer_comment": {"type": "string"},
                                "discussion_summary": {"type": "string"},
                            },
                            "required": ["first_mention_timestamp", "original_reviewer_comment", "discussion_summary"],
                        },
                        "severity_assessment": {
                            "type": "object",
                            "properties": {
                                "justification": {"type": "string"},
                                "severity_level": {"type": "integer"},
                            },
                            "required": ["justification", "severity_level"],
                        },
                        "resolving_information": {
                            "type": "object",
                            "properties": {
                                "commit_sha": {"type": "string"},
                                "code_snippet": {"type": "string"},
                                "resolution_explanation": {"type": "string"},
                            },
                            "additionalProperties": False
                        }
                    },
                    "required": ["change_type", "change_introduction", "discussion_evidence", "severity_assessment"],
                    "additionalProperties": False
                }
            }
        }
    }

    return prompt, response_format


def change_analysis(prompt: str, response_format: str) -> list[dict]:
    """
    Parses the raw text output from the change/defect analysis LLM, extracts structured 
    change/defect information into a JSON list, validates each change/defect object, and 
    returns a list of validated changes/defects.

    Args:
        analysis_llm_answer: The raw string response from the analysis LLM 
                             (generated by construct_merged_prompt).

    Returns:
        A list of dictionaries, where each dictionary represents a validated 
        defect object. Returns an empty list if no valid defects are found, 
        or if the input indicates no defects were identified.
    """
    
    # --- 3. Call Parser LLM ---
    try:
        json_str = run_chat(CHAT_MODEL, [{"role": "user", "content": prompt}], temperature=OPENAI_TEMPERATURE, response_format=response_format)
    except Exception as e:
        logging.error(f"Error calling parser LLM: {e}")
        return None # Return empty list on LLM call failure
    
    # --- 5. Parse JSON String ---
    try:
        parsed_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}. Raw JSON string: '{json_str}'")
        return None # Return empty list if JSON is malformed

    # --- 6. Validate Extracted Data ---
    validated_changes = []
    
    # Ensure parsed_data is a list, even if LLM returned single object
    if not isinstance(parsed_data, list):
       logging.error(f"Expected JSON list, but got {type(parsed_data)}. Data: {parsed_data}")
       return []

    for index, change_obj in enumerate(parsed_data):
        is_valid, change_code = validate_single_change_object(change_obj, index)
        if is_valid:
             # Optional: Add the extracted change code if needed downstream
             # change_obj['change_code_extracted'] = change_code 
             validated_changes.append(change_obj)
        else:
            raise Exception(f"Change object at index {index} failed validation. Object: {json.dumps(change_obj, indent=2)}")
            
    # --- 7. Return Validated Defects ---
    if not validated_changes:
        logging.info("Parsing complete, but no valid change objects were found after validation.")
    else:
        logging.info(f"Successfully parsed and validated {len(validated_changes)} change object(s).")
        
    return validated_changes

# --- Helper: Validation Logic ---
def validate_single_change_object(change_obj: dict, index: int) -> tuple[bool, str | None]:
    """Validates a single change dictionary extracted by the parser LLM."""
    
    # --- A. Extract Defect Type/Code ---
    change_type_str = change_obj.get("change_type")
    if not change_type_str or not isinstance(change_type_str, str):
        logging.error(f"Validation Failed (Index {index}): Missing or invalid 'Change Type'.")
        return False, None
        
    # Extract the code (e.g., "F.6", "E.1.1") from the classification string
    match = re.match(r"([EF]\.\d(\.\d)?)", change_type_str)
    if not match:
        logging.error(f"Validation Failed (Index {index}): Could not extract change code from Classification '{change_type_str}'.")
        return False, None
    change_code = match.group(1)
    is_f6 = (change_code == "F.6")

    # --- B. Define Required Paths based on change_code ---
    # Using dot notation for easy checking
    required_paths = [
        "change_type", # Already checked, but good to list
        "change_introduction.commit_sha",
        "change_introduction.code_snippet",
        "discussion_evidence.first_mention_timestamp",
        "discussion_evidence.original_reviewer_comment",
        "discussion_evidence.discussion_summary",
        "severity_assessment.justification",
        "severity_assessment.severity_level",
    ]
    
    if not is_f6:
        required_paths.extend([
            "resolving_information.commit_sha",
            "resolving_information.code_snippet",
            "resolving_information.resolution_explanation",
        ])
    else:
        # F.6 should NOT have Resolving Information
        if "resolving_information" in change_obj:
             logging.error(f"Validation Failed (Index {index}): 'resolving_information' should NOT be present for F.6 defect.")
             return False, change_code

    # --- C. Check Presence and Basic Type of Required Fields ---
    for path in required_paths:
        keys = path.split('.')
        current_level = change_obj
        valid_path = True
        for i, key in enumerate(keys):
            if not isinstance(current_level, dict) or key not in current_level:
                logging.error(f"Validation Failed (Index {index}): Missing required path element: '{path}' (missing key: '{key}').")
                valid_path = False
                break
            if i < len(keys) - 1: # If not the last key, move down
                 current_level = current_level[key]
            else: # Last key, check type (can add more specific type checks if needed)
                value = current_level[key]
                if value is None or (isinstance(value, str) and not value.strip()) : # Basic check for empty/null
                     logging.warning(f"Validation Warning (Index {index}): Field '{path}' is present but empty or None.")
                     # Decide if empty strings are acceptable or should fail validation
                     # For now, just warn. Could return False here if needed.
                
        if not valid_path:
            return False, change_code # Stop validation if path is broken


    # --- D. Check Specific Field Constraints ---
    # Severity Level
    try:
        severity_level = change_obj.get("severity_assessment").get("severity_level")
        if severity_level is None:
             raise ValueError("Severity Level is missing") # Should have been caught by path check, but belt-and-suspenders
        
        severity_int = int(severity_level) # Check if it can be cast to int
        if not 1 <= severity_int <= 10:
            logging.error(f"Validation Failed (Index {index}): 'severity_assessment.severity_level' ({severity_level}) must be between 1 and 10.")
            return False, change_code
    except (ValueError, TypeError) as e:
        logging.error(f"Validation Failed (Index {index}): 'severity_assessment.severity_level' ({severity_level}) is not a valid integer. Error: {e}")
        return False, change_code

    # Add more specific checks if needed (e.g., SHA format, timestamp format)

    # If all checks pass
    return True, change_code


def check_external_url(timeline):
    for event in timeline:
        if event['type'] in ['description', 'comment', 'review_comment', 'review']:
            if 'https://' in event['body'] or 'http://' in event['body']:
                return True
    return False


def process_pr(pr_info):
    try:
        logging.info(f"开始处理PR #{pr_info['number']}")
        result = {
            'repo': pr_info['repo'],
            'pr_number': pr_info['number'],
            'pr_title': pr_info['title'],
            'user': pr_info['user'],
            'url': pr_info['url'],
            'created_at': pr_info['created_at'],
            'merged_at': pr_info['merged_at'],
            'base_commit': pr_info['base_commit'],
            'defect_introduced': False,
            'init_prompt': None,
            'init_answer': None,
            'timeline': pr_info['timeline'],
            'defects': []
        }
        
        # if check_external_url(pr_info['timeline']):
        #     logging.info(f"PR #{pr_info['number']}: 存在外部URL，跳过")
        #     return result  # 返回 None 表示处理失败
        
        # 初始提问
        logging.info(f"PR #{pr_info['number']}: 构建初始提问")
        init_prompt, response_format = construct_merged_prompt(pr_info)
        result['init_prompt'] = init_prompt
        
        if len(init_prompt) > 100000:
            logging.info(f"PR #{pr_info['number']}: 初始提问过长，跳过")
            return result
        
        try:
            logging.info(f"PR #{pr_info['number']}: 发送初始提问到 { CHAT_MODEL } 模型")
            parsed = change_analysis(init_prompt, response_format)
            result['init_answer'] = parsed            
            if parsed is None:
                logging.error(f"PR #{pr_info['number']}: 解析初始回答失败")
                return None  # 返回 None 表示处理失败
        except Exception as e:
            logging.error(f"PR #{pr_info['number']}: 解析初始回答时发生错误: {str(e)}")
            return None  # 返回 None 表示处理失败
        
        result.update({
            'defect_introduced': len(parsed) > 0,
            'defects': parsed
        })
        
        logging.info(f"PR #{pr_info['number']}: 处理完成")
        return result

    except Exception as e:
        logging.error(f"处理PR #{pr_info['number']} 失败: {e.__class__.__name__}: {str(e)}")
        return None

def is_valid_sha(sha):
    # 检查是否为有效的 SHA-1 值（40个十六进制字符）
    return isinstance(sha, str) and len(sha) == 40 and all(c in '0123456789abcdefABCDEF' for c in sha)

def collect_repo_prs(repo_name, result_path):
    # 设置日志
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(result_path, 'pr_analysis.log')),
            logging.StreamHandler()
        ]
    )

    logging.info(f"启动分析流程，仓库: {repo_name}，路径: {result_path}")

    # Get PRs, using cache if available
    prs = load_jsonl(f'data_train/prs_raw/{repo_name.replace("/", "__")}__prs.jsonl')
    # pr_ids_range = (5000, 15000)
    pr_ids_range = None
    filtered_prs = filter_prs(repo_name, prs, max_commits=10, pr_ids_range=pr_ids_range)
    results = []
    error_count = 0
    fixed_error_prs = []

    analysised_prs_keys = set()
    os.makedirs(result_path, exist_ok=True)
    if os.path.exists(os.path.join(result_path, 'analysis_results.jsonl')):
        with open(os.path.join(result_path, 'analysis_results.jsonl'), 'r', encoding='utf-8') as f:
            for line in f:
                pr = json.loads(line)
                analysised_prs_keys.add(f"pr_{pr['pr_number']}_{pr['pr_title']}")
                results.append(pr)
                if pr['defect_introduced']:
                    fixed_error_prs.append(pr['pr_number'])
                
    logging.info(f"已分析 {len(analysised_prs_keys)} 个PR")
    filtered_prs = [pr for pr in filtered_prs if f"pr_{pr['number']}_{pr['title']}" not in analysised_prs_keys]
    logging.info(f"还需要分析 {len(filtered_prs)} 个PR")

    # for pr in tqdm(filtered_prs, total=len(filtered_prs), desc="处理PR"):
    #     # if pr.number != 10893:
    #     #     continue
    #     # import pdb; pdb.set_trace()
    #     result = process_pr(pr)
    
    file_lock = threading.Lock()
    def process_pr_worker(pr):
        result = process_pr(pr)
        if result is not None:
            with file_lock:
                with open(os.path.join(result_path, 'analysis_results.jsonl'), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')
            result = {
                'repo': result['repo'],
                'pr_number': result['pr_number'],
                'pr_title': result['pr_title'],
                'defect_introduced': result['defect_introduced'],
            }
        return result

    with ThreadPoolExecutor(max_workers=16) as executor:
        for result in tqdm(executor.map(process_pr_worker, filtered_prs), total=len(filtered_prs), desc="处理PR"):
            if result is None:
                error_count += 1
            else:
                results.append(result)
                if result['defect_introduced']:
                    fixed_error_prs.append(result['pr_number'])
    
    logging.info(f"分析完成，结果已保存到 analysis_results.jsonl")
    logging.info(f"总共处理 {len(prs)} 个PR")
    logging.info(f"其中需要分析{len(filtered_prs)}个PR")
    logging.info(f"成功处理 {len(results)} 个PR")
    logging.info(f"处理失败 {error_count} 个PR")
    logging.info(f"发现并修复错误的PR数量: {len(fixed_error_prs)}")

    # if fixed_error_prs:
    #     logging.info("以下PR发现并修复了错误:")
    #     for pr_number in fixed_error_prs:
    #         logging.info(f"PR #{pr_number}")
    # else:
    #     logging.info("没有发现引入并修复错误的PR")

    #这里使用w也没事，反正之前存储过了id
    # 将发现并修复错误的PR编号保存到文件
    with open(os.path.join(result_path, 'fixed_error_prs.txt'), 'w') as f:
        for pr_number in fixed_error_prs:
            f.write(f"{pr_number}\n")
    logging.info("发现并修复错误的PR编号已保存到 fixed_error_prs.txt")


def verify_defects(pr_info, pr_info_list, static_metrics):
    pr_number = pr_info['pr_number']
    
    # 多次运行结果一致性检查的要求可能太高了，导致很多PR被过滤掉，用严重性筛选可能更好
    # # 检查defect_types是否一致
    # same_defect_types = True
    # defect_types = [result['defect_type'] for result in pr_info_list]
    # for defect_type in defect_types:
    #     defect_type.sort()
    #     if defect_type != defect_types[0]:
    #         same_defect_types = False
    #         break
    # if not same_defect_types:
    #     logging.warning(f"PR #{pr_number} 存在多个不同的 defect_types 分析结果")
    #     static_metrics["defect_type_filter"].append(f"{pr_number}, {', '.join([str(defect_type) for defect_type in defect_types])}")
    #     return False
        
    # # 检查intrducing_commit是否一致
    # same_intrducing_commit = True
    # intrducing_commits = [[defect['defect_introducing_commit']['sha'] for defect in result['defects']] for result in pr_info_list]
    # for intrducing_commit in intrducing_commits:
    #     intrducing_commit.sort()
    #     if intrducing_commit != intrducing_commits[0]:
    #         same_intrducing_commit = False
    #         break
    # if not same_intrducing_commit:
    #     logging.warning(f"PR #{pr_number} 存在多个不同的 intrducing_commit 分析结果")
    #     static_metrics["intrducing_commit_diff_filter"].append(pr_number)
    #     return False
    
    # # 检查fix_commits是否一致
    # same_fix_commits = True
    # all_fix_commits = []
    # for pr_info in pr_info_list:
    #     pr_fix_commits = []
    #     for defect in pr_info['defects']:
    #         for fix_commit in defect['fix_commits']:
    #             pr_fix_commits.append(fix_commit['sha'])
    #     all_fix_commits.append(pr_fix_commits)
        
    # for fix_commits in all_fix_commits:
    #     fix_commits.sort()
    #     if fix_commits != all_fix_commits[0]:
    #         same_fix_commits = False
    #         break
    # if not same_fix_commits:
    #     logging.warning(f"PR #{pr_number} 存在多个不同的 fix_commits 分析结果")
    #     static_metrics["fix_commits_diff_filter"].append(pr_number)
    #     return False
    
    all_commits = [e for e in pr_info['timeline'] if e['type'] == 'commit']
    for defect in pr_info['defects']:
        defect_commit = [commit for commit in all_commits if commit['sha'] == defect['change_introduction']['commit_sha']]
        if len(defect_commit) == 0:
            logging.warning(f"Defect commit {defect['change_introduction']['commit_sha']} not found for {pr_info['pr_number']}")
            static_metrics["defect_introducing_commit_count_filter"].append(pr_info['pr_number'])
            return False
        defect_commit = defect_commit[0]
        
        if safe_parse_time(defect_commit['date']) > safe_parse_time(pr_info['created_at']):
            logging.warning(f"Defect commit {defect_commit['sha']} is after PR creation for {pr_info['pr_number']}")
            static_metrics["defect_introducing_commit_time_filter"].append(pr_info['pr_number'])
            return False
        
        # import pdb; pdb.set_trace()
        try:
            if safe_parse_time(pr_info['created_at']) > safe_parse_time(defect['discussion_evidence']['first_mention_timestamp']) or \
                safe_parse_time(defect_commit['date']) > safe_parse_time(defect['discussion_evidence']['first_mention_timestamp']):
                logging.warning(f"Defect commit {defect_commit['sha']} is after discussion evidence for {pr_info['pr_number']}")
                static_metrics["defect_discussion_time_filter"].append(pr_info['pr_number'])
                return False
        except Exception as e:
            logging.warning(f"Parse time error {defect['discussion_evidence']['first_mention_timestamp']} for {pr_info['pr_number']}")
            static_metrics["defect_discussion_time_filter"].append(pr_info['pr_number'])
            return False
        
        if 'resolving_information' in defect:
            if defect['change_introduction']['commit_sha'] == defect['resolving_information']['commit_sha']:
                logging.warning(f"PR #{pr_number} 存在相同的 change_introduction 和 resolving_information")
                static_metrics["same_commit_filter"].append(pr_number)
                return False
        
            fix_commit_info = [commit for commit in all_commits if commit['sha'] == defect['resolving_information']['commit_sha']]
            if len(fix_commit_info) == 0:
                logging.warning(f"Fix commit {defect['resolving_information']['commit_sha']} not found for {pr_info['pr_number']}")
                static_metrics["fix_commit_count_filter"].append(pr_info['pr_number'])
                return False
        
            fix_commit_info = fix_commit_info[0]
            try:
                if safe_parse_time(fix_commit_info['date']) < safe_parse_time(defect['discussion_evidence']['first_mention_timestamp']):
                    logging.warning(f"Fix commit {fix_commit_info['sha']} is before defect discussion for {pr_info['pr_number']}")
                    static_metrics["fix_commit_time_filter"].append(pr_info['pr_number'])
                    return False
            except Exception as e:
                logging.warning(f"Parse time error {defect['discussion_evidence']['first_mention_timestamp']} for {pr_info['pr_number']}")
                static_metrics["fix_commit_time_filter"].append(pr_info['pr_number'])
                return False
            
    return True

def verify_defect_prs(repo_name, result_paths, save_dir):
    logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(message)s',
                   handlers=[
                       logging.FileHandler(os.path.join(save_dir, 'verify_prs.log')),
                       logging.StreamHandler()
                   ])
    logging.info("开始验证分析结果")
    static_metrics = {"defect_introduced_filter":[], "defect_type_filter":[], "intrducing_commit_filter":[],
                      "pr_description_filter":[], "defect_verify_filter":[], "pr_count_filter":[], "pr_commit_filter":[],
                      "defect_count_filter":[], "intrducing_commit_diff_filter":[], "fix_commits_diff_filter":[],
                      "defect_introducing_commit_count_filter":[], "defect_introducing_commit_time_filter":[],
                      "fix_commit_count_filter":[], "fix_commit_time_filter":[], "f6_merged_filter":[], "f6_not_merged_filter":[],
                      "defect_discussion_time_filter":[], "same_commit_filter":[]}
    raw_results = {}
    for result_path in result_paths:
        # 读取analysis_results.jsonl
        with open(os.path.join(result_path, 'analysis_results.jsonl'), 'r', encoding='utf-8') as f:
            one_results = [json.loads(line) for line in f]
        for r in one_results:
            if r['pr_number'] not in raw_results:
                raw_results[r['pr_number']] = []
            raw_results[r['pr_number']].append(r)
        static_metrics[result_path] = {
            'total_prs': len(one_results),
            'defect_introduced_prs': len([r for r in one_results if r['defect_introduced']]),
            'clean_prs': len([r for r in one_results if r['defect_introduced']==False]),
        }
    
    verified_results = []
    for pr_number, results in raw_results.items():
        if len(results) != len(result_paths):
            logging.warning(f"PR #{pr_number} 的分析结果数量与运行次数不一致")
            static_metrics["pr_count_filter"].append(pr_number)
            continue
        
        if any(r['defect_introduced']==False for r in results):
            if any(r['defect_introduced']==True for r in results):
                logging.warning(f"PR #{pr_number} 非多次确认引入错误")
                static_metrics["defect_introduced_filter"].append(pr_number)
            continue
        
        max_defect_idx = 0
        max_defect_count = 0
        for i, r in enumerate(results):
            if len(r['defects']) > max_defect_count:
                max_defect_count = len(r['defects'])
                max_defect_idx = i
        pr_info = results[max_defect_idx]
        
        pr_description = [e for e in pr_info['timeline'] if e['type'] == 'description']
        if len(pr_description) == 0:
            static_metrics["pr_description_filter"].append(pr_number)
            logging.warning(f"PR #{pr_number} 没有找到 pr_description")
            continue
        
        empty_commits = [e for e in pr_info['timeline'] if e['type'] == 'commit' and len(e['diff']) == 0]
        if len(empty_commits) > 0:
            static_metrics["pr_commit_filter"].append(pr_number)
            logging.warning(f"PR #{pr_number} 存在空提交")
            continue
        
        if len(pr_info['defects']) == 0:
            static_metrics["defect_count_filter"].append(pr_number)
            logging.warning(f"PR #{pr_number} 没有找到 defects")
            continue
        
        defect_types = [defect['change_type'].split(' ')[0] for defect in pr_info['defects']]
        
        if pr_info['merged_at'] is None and not "F.6" in defect_types:
            static_metrics["f6_not_merged_filter"].append(pr_number)
            logging.warning(f"PR #{pr_number} 未合并，且不是F.6")
            continue
        if "F.6" in defect_types and pr_info['merged_at'] is not None:
            static_metrics["f6_merged_filter"].append(pr_number)
            logging.warning(f"PR #{pr_number} 是F.6，但已合并")
            continue
        
        if not verify_defects(pr_info, results, static_metrics):
            continue
        pr_info['repo'] = repo_name
        severity_list = [defect['severity_assessment']['severity_level'] for defect in pr_info['defects']]
        verified_results.append({'severity': np.mean(severity_list).round(2), **pr_info})

    static_metrics["verified_prs"] = len(verified_results)
    for key in list(static_metrics.keys()):
        if key.endswith("filter"):
            static_metrics[f"{key}_count"] = len(static_metrics[key])
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'verified_defect_results.jsonl'), 'w', encoding='utf-8') as f:
        for result in verified_results:
            f.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')
    with open(os.path.join(save_dir, 'verified_defect_results.json'), 'w', encoding='utf-8') as f:
        json.dump(verified_results, f, ensure_ascii=False, indent=4)
    with open(os.path.join(save_dir, 'verified_defect_static_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(static_metrics, f, ensure_ascii=False, indent=4)
    # 报告验证结果
    logging.info(f"验证完成，结果已保存到 {save_dir}/verified_defect_results.jsonl")
    logging.info(f"被确认为引入并修复错误的PR数量: {len(verified_results)}")
    return verified_results

def verify_clean_prs(repo_name, result_paths, save_dir):
    logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(message)s',
                   handlers=[
                       logging.FileHandler(os.path.join(save_dir, 'verify_prs.log')),
                       logging.StreamHandler()
                   ])
    logging.info("开始验证分析结果")
    static_metrics = {"defect_introduced_filter":[], "defect_type_filter":[], "intrducing_commit_filter":[],
                      "pr_description_filter":[], "defect_verify_filter":[], "pr_count_filter":[], "pr_commit_filter":[],
                      "defect_count_filter":[], "intrducing_commit_diff_filter":[], "fix_commits_diff_filter":[],
                      "defect_introducing_commit_count_filter":[], "defect_introducing_commit_time_filter":[],
                      "fix_commit_count_filter":[], "fix_commit_time_filter":[], "f6_merged_filter":[], "f6_not_merged_filter":[],
                      "init_answer_filter":[]}
    raw_results = {}
    for result_path in result_paths:
        # 读取analysis_results.jsonl
        with open(os.path.join(result_path, 'analysis_results.jsonl'), 'r', encoding='utf-8') as f:
            one_results = [json.loads(line) for line in f]
        for r in one_results:
            if r['pr_number'] not in raw_results:
                raw_results[r['pr_number']] = []
            raw_results[r['pr_number']].append(r)
        static_metrics[result_path] = {
            'total_prs': len(one_results),
            'defect_introduced_prs': len([r for r in one_results if r['defect_introduced']]),
            'clean_prs': len([r for r in one_results if r['defect_introduced']==False]),
        }
    
    verified_results = []
    for pr_number, results in raw_results.items():
        if len(results) != len(result_paths):
            static_metrics["pr_count_filter"].append(pr_number)
            continue
        
        if any(r['defect_introduced']==True for r in results):
            static_metrics["defect_introduced_filter"].append(pr_number)
            continue

        pr_info = results[0]
        
        if pr_info['init_answer'] is None:
            static_metrics["init_answer_filter"].append(pr_number)
            continue
        
        if pr_info['merged_at'] is None:
            static_metrics["f6_not_merged_filter"].append(pr_number)
            continue
        
        pr_description = [e for e in pr_info['timeline'] if e['type'] == 'description']
        if len(pr_description) == 0:
            static_metrics["pr_description_filter"].append(pr_number)
            continue
        
        empty_commits = [e for e in pr_info['timeline'] if e['type'] == 'commit' and len(e['diff']) == 0]
        if len(empty_commits) > 0:
            static_metrics["pr_commit_filter"].append(pr_number)
            continue
        
        all_commits = [e for e in pr_info['timeline'] if e['type'] == 'commit']
        instance_commits = []
        for commit in all_commits:
            if safe_parse_time(commit['date']) <= safe_parse_time(pr_info['created_at']):
                instance_commits.append(commit)
        instance_commits.sort(key=lambda x: safe_parse_time(x['date']))

        # Check if instance_commits is empty before proceeding
        if len(instance_commits) == 0:
            static_metrics["pr_commit_filter"].append(pr_number)
            continue
        
        pr_info['repo'] = repo_name
        verified_results.append(pr_info)

    static_metrics["verified_prs"] = len(verified_results)
    for key in list(static_metrics.keys()):
        if key.endswith("filter"):
            static_metrics[f"{key}_count"] = len(static_metrics[key])
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'verified_clean_results.jsonl'), 'w', encoding='utf-8') as f:
        for result in verified_results:
            f.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')
    with open(os.path.join(save_dir, 'verified_clean_results.json'), 'w', encoding='utf-8') as f:
        json.dump(verified_results, f, ensure_ascii=False, indent=4)
    with open(os.path.join(save_dir, 'verified_clean_static_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(static_metrics, f, ensure_ascii=False, indent=4)
    # 报告验证结果
    logging.info(f"验证完成，结果已保存到 {save_dir}/verified_clean_results.jsonl")
    logging.info(f"被确认为无错误的PR数量: {len(verified_results)}")
    return verified_results

def analyse_prs(data_paths, save_dir, save_suffix=""):
    static_metrics = {}

    for data_path in data_paths:
        one_results = load_jsonl(data_path)
        ont_static_metrics = {
            "defect_count": {},
            "defect_type_all": {},
            "defect_type_one": {},
            "verified_prs": 0,
        }
        for pr_info in one_results:
            if not pr_info['defect_introduced']:
                continue
            ont_static_metrics["verified_prs"] += 1
            ont_static_metrics["defect_count"][len(pr_info['defects'])] = ont_static_metrics["defect_count"].get(len(pr_info['defects']), 0) + 1
            change_types = [defect['change_type'].split(' ')[0] for defect in pr_info['defects']]
            for change_type in change_types:
                ont_static_metrics["defect_type_all"][change_type] = ont_static_metrics["defect_type_all"].get(change_type, 0) + 1
            
            for change_type in set(change_types):
                ont_static_metrics["defect_type_one"][change_type] = ont_static_metrics["defect_type_one"].get(change_type, 0) + 1
            
        static_metrics[data_path] = ont_static_metrics
        
    print(static_metrics)
    print(f"分析完成，结果已保存到 {save_dir}/analyse_metrics{save_suffix}.json")
    with open(os.path.join(save_dir, f'analyse_metrics{save_suffix}.json'), 'w', encoding='utf-8') as f:
        json.dump(static_metrics, f, ensure_ascii=False, indent=4)


def fliter_by_szz(save_dir, verified_results):
    szz_results_path = os.path.join(save_dir, 'szz_results.json')
    if not os.path.exists(szz_results_path):
        logging.warning(f"SZZ 结果文件 {szz_results_path} 不存在")
        szz_results = []
    else:
        szz_results = load_json(szz_results_path)
    inducing_fix_commits = {}
    for szz_result in szz_results:
        for inducing_commit_hash in szz_result['inducing_commit_hash']:
            if inducing_commit_hash not in inducing_fix_commits:
                inducing_fix_commits[inducing_commit_hash] = []
            inducing_fix_commits[inducing_commit_hash].append(szz_result['fix_commit_hash'])
    
    filtered_prs = []
    for pr_info in verified_results:
        commits = [e['sha'] for e in pr_info['timeline'] if e['type'] == 'commit']
        for commit in commits:
            if commit in inducing_fix_commits:
                if not all([fix_commit in commits for fix_commit in inducing_fix_commits[commit]]):
                    filtered_prs.append(pr_info['pr_number'])
    
    save_prs = [pr_info for pr_info in verified_results if pr_info['pr_number'] not in filtered_prs]
    logging.info(f"SZZ 过滤掉 {len(filtered_prs)} 个PR")
    return save_prs

def fliter_by_gitlog(save_dir, verified_results):
    gitlog_results_path = '/SWRBench/data/check_log_not_success_results.json'
    if not os.path.exists(gitlog_results_path):
        logging.warning(f"gitlog 结果文件 {gitlog_results_path} 不存在")
        gitlog_results = []
    else:
        gitlog_results = load_json(gitlog_results_path)

    filtered_prs = []
    for pr_info in verified_results:
        instance_id = f"{pr_info['repo'].replace('/', '__')}-{pr_info['pr_number']}"
        if instance_id in gitlog_results:
            filtered_prs.append(pr_info['pr_number'])
    
    save_prs = [pr_info for pr_info in verified_results if pr_info['pr_number'] not in filtered_prs]
    logging.info(f"gitlog 过滤掉 {len(filtered_prs)} 个PR")
    return save_prs

def fliter_by_severity(save_dir, original_prs):
    # save_prs = [pr_info for pr_info in original_prs if pr_info['severity'] > 3]
    save_prs = []
    for pr_info in original_prs:
        max_severity = max([defect['severity_assessment']['severity_level'] for defect in pr_info['defects']])
        if max_severity >= 3:
            save_prs.append(pr_info)
    logging.info(f"严重性过滤掉 {len(original_prs) - len(save_prs)} 个PR")
    return save_prs


def sample_prs(save_dir, original_prs):
    random.seed(42)
    # selected_prs = original_prs
    # return selected_prs
    
    type2prs = {}
    for pr_info in original_prs:
        for defect in pr_info['defects']:
            if defect['change_type'].split(' ')[0] not in type2prs:
                type2prs[defect['change_type'].split(' ')[0]] = []
            type2prs[defect['change_type'].split(' ')[0]].append(pr_info['pr_number'])
            
    sample_count = len(type2prs['F.2']) if 'F.2' in type2prs else 1
    logging.info(f"样本选择数 F.2 为 {sample_count}")
    selected_prs_numbers = []
    for dtype, prs in type2prs.items():
        sample_prs_data = random.sample(prs, min(len(prs), sample_count))
        selected_prs_numbers.extend(sample_prs_data)
    
    selected_prs_numbers = list(set(selected_prs_numbers))
    selected_prs = [pr_info for pr_info in original_prs if pr_info['pr_number'] in selected_prs_numbers]
    logging.info(f"样本选择完成，从 {len(original_prs)} 个PR中选择 {len(selected_prs)} 个PR")
    return selected_prs


def sample_clean_prs(save_dir, defect_prs, clean_prs):
    logging.info(f"Starting TWO-STAGE stratified sampling: {len(defect_prs)} defect PRs, {len(clean_prs)} available clean PRs.")
    random.seed(42)

    # --- Pre-calculation ---
    if not defect_prs:
        logging.warning("No defect PRs provided. Cannot sample.")
        return []
    if not clean_prs:
        logging.warning("No clean PRs available to sample from.")
        return []

    # Calculate metrics for ALL defect PRs
    defect_metrics = []
    for pr_info in defect_prs:
        pr_commits = [e for e in pr_info['timeline'] if e['type'] == 'commit' and safe_parse_time(e['date']) <= safe_parse_time(pr_info['created_at'])]
        count = len(pr_commits)
        length = sum(len(e.get('diff_text', '')) for e in pr_commits)
        log_length = np.log1p(length)
        defect_metrics.append({
            'pr_number': pr_info['pr_number'], 'commit_count': count,
            'commit_length': length, 'log_commit_length': log_length
        })
    defect_df = pd.DataFrame(defect_metrics)
    num_defect_prs = len(defect_df)

    # Calculate metrics for ALL clean PRs
    clean_metrics = []
    pr_map = {pr['pr_number']: pr for pr in clean_prs}
    for pr_info in clean_prs:
        pr_commits = [e for e in pr_info['timeline'] if e['type'] == 'commit' and safe_parse_time(e['date']) <= safe_parse_time(pr_info['created_at'])]
        count = len(pr_commits)
        length = sum(len(e.get('diff_text', '')) for e in pr_commits)
        log_length = np.log1p(length)
        clean_metrics.append({
            'pr_number': pr_info['pr_number'], 'commit_count': count,
            'commit_length': length, 'log_commit_length': log_length
        })
    clean_df = pd.DataFrame(clean_metrics)
    clean_df.set_index('pr_number', inplace=True, drop=False)

    # --- Stage 1: Sample based on Commit Count ---
    logging.info(f"--- Stage 1: Sampling based on Commit Count ---")
    oversample_factor = 2.0
    intermediate_target_size = min(int(num_defect_prs * oversample_factor), len(clean_df))
    logging.info(f"Targeting intermediate size: {intermediate_target_size}")

    n_bins_count = 4 # Bins for commit count
    try:
        _, count_bin_edges = pd.qcut(defect_df['commit_count'], q=n_bins_count, retbins=True, duplicates='drop')
        count_bin_edges[0] = -np.inf
        count_bin_edges[-1] = np.inf
    except ValueError as e:
        logging.warning(f"Could not create commit_count quantiles for Stage 1: {e}. Falling back to random sampling for intermediate set.")
        intermediate_pr_numbers = random.sample(clean_df['pr_number'].tolist(), intermediate_target_size)
        intermediate_clean_df = clean_df.loc[intermediate_pr_numbers].copy() # Create copy for stage 2
        # Go directly to stage 2
        num_defect_prs = len(defect_df) # Needed for stage 2 target
        logging.info(f"(Fallback) Randomly selected {len(intermediate_clean_df)} PRs for intermediate set.")
        # Continue to Stage 2 below...

    else: # If binning worked
        # Assign count strata to defect PRs (to get target proportions)
        defect_df['count_stratum'] = pd.cut(defect_df['commit_count'], bins=count_bin_edges, labels=False, include_lowest=True)
        defect_count_stratum_counts = defect_df['count_stratum'].value_counts().to_dict()

        # Assign count strata to clean PRs
        clean_df['count_stratum'] = pd.cut(clean_df['commit_count'], bins=count_bin_edges, labels=False, include_lowest=True)
        available_clean_by_count_stratum = clean_df.groupby('count_stratum')

        intermediate_pr_numbers = []
        sampled_indices_stage1 = set()

        logging.info("Sampling proportionally within count strata for intermediate set...")
        for stratum, defect_count in defect_count_stratum_counts.items():
            target_proportion = defect_count / num_defect_prs
            stratum_target_size = int(round(target_proportion * intermediate_target_size))

            if stratum in available_clean_by_count_stratum.groups:
                stratum_clean_prs_df = available_clean_by_count_stratum.get_group(stratum)
                stratum_indices = stratum_clean_prs_df.index.tolist()

                num_to_sample = min(stratum_target_size, len(stratum_indices))
                sampled_now = random.sample(stratum_indices, num_to_sample)
                intermediate_pr_numbers.extend(sampled_now)
                sampled_indices_stage1.update(sampled_now)
            else:
                logging.debug(f"Stage 1: Count Stratum {stratum}: Target={stratum_target_size}, No clean PRs.")

        # Handle deficit for intermediate set (if sum of samples < intermediate_target_size)
        current_intermediate_size = len(intermediate_pr_numbers)
        deficit_intermediate = intermediate_target_size - current_intermediate_size
        if deficit_intermediate > 0:
            logging.info(f"Stage 1 deficit: {deficit_intermediate}. Sampling randomly from remaining.")
            remaining_clean_indices = list(set(clean_df.index) - sampled_indices_stage1)
            num_to_sample_deficit = min(deficit_intermediate, len(remaining_clean_indices))
            if num_to_sample_deficit > 0:
                additional_indices = random.sample(remaining_clean_indices, num_to_sample_deficit)
                intermediate_pr_numbers.extend(additional_indices)
                sampled_indices_stage1.update(additional_indices)

        # Ensure uniqueness for intermediate set
        intermediate_pr_numbers = list(set(intermediate_pr_numbers))
        # Get the DataFrame for the intermediate set
        intermediate_clean_df = clean_df.loc[intermediate_pr_numbers].copy() # Create copy for stage 2
        logging.info(f"Stage 1 sampling complete. Intermediate set size: {len(intermediate_clean_df)}")


    # --- Stage 2: Sample based on Log Commit Length (from Intermediate Set) ---
    logging.info(f"--- Stage 2: Sampling based on Log Commit Length from Intermediate Set ---")
    final_target_size = num_defect_prs
    logging.info(f"Targeting final size: {final_target_size} from intermediate set of {len(intermediate_clean_df)}")

    if len(intermediate_clean_df) < final_target_size:
        logging.warning(f"Intermediate set ({len(intermediate_clean_df)}) is smaller than final target ({final_target_size}). Returning all intermediate PRs.")
        selected_prs = [pr_map[num] for num in intermediate_clean_df.index if num in pr_map]
        return selected_prs
    if len(intermediate_clean_df) == 0:
         logging.warning("Intermediate set is empty. Cannot perform Stage 2.")
         return []


    n_bins_log_length = 4 # Bins for log commit length
    try:
        # Define bins based on ORIGINAL defect set's log length
        _, log_length_bin_edges = pd.qcut(defect_df['log_commit_length'], q=n_bins_log_length, retbins=True, duplicates='drop')
        log_length_bin_edges[0] = -np.inf
        log_length_bin_edges[-1] = np.inf
    except ValueError as e:
        logging.warning(f"Could not create log_commit_length quantiles for Stage 2: {e}. Falling back to random sampling from intermediate set.")
        final_selected_pr_numbers = random.sample(intermediate_clean_df.index.tolist(), final_target_size)
        selected_prs = [pr_map[num] for num in final_selected_pr_numbers if num in pr_map]
        logging.info(f"(Fallback) Randomly selected {len(selected_prs)} PRs from intermediate set.")
        # Skip verification as distributions won't match well
        return selected_prs

    # Assign log-length strata to defect PRs (to get target counts for final set)
    defect_df['log_length_stratum'] = pd.cut(defect_df['log_commit_length'], bins=log_length_bin_edges, labels=False, include_lowest=True)
    defect_log_length_stratum_counts = defect_df['log_length_stratum'].value_counts().to_dict()

    # Assign log-length strata to the INTERMEDIATE clean PRs
    intermediate_clean_df['log_length_stratum'] = pd.cut(intermediate_clean_df['log_commit_length'], bins=log_length_bin_edges, labels=False, include_lowest=True)
    available_intermediate_by_log_length = intermediate_clean_df.groupby('log_length_stratum')

    final_selected_pr_numbers = []
    sampled_indices_stage2 = set()

    logging.info("Sampling within log-length strata from intermediate set...")
    # Sample exactly the target count for each log-length stratum based on defect distribution
    for stratum, target_count in defect_log_length_stratum_counts.items():
        if stratum in available_intermediate_by_log_length.groups:
            stratum_intermediate_prs_df = available_intermediate_by_log_length.get_group(stratum)
            stratum_indices = stratum_intermediate_prs_df.index.tolist()

            num_to_sample = min(target_count, len(stratum_indices)) # Sample up to target or available
            sampled_now = random.sample(stratum_indices, num_to_sample)
            final_selected_pr_numbers.extend(sampled_now)
            sampled_indices_stage2.update(sampled_now)
            if num_to_sample < target_count:
                 logging.debug(f"Stage 2: Log-Length Stratum {stratum}: Target={target_count}, Available={len(stratum_indices)}. Sampled all available.")
        else:
             logging.debug(f"Stage 2: Log-Length Stratum {stratum}: Target={target_count}, No PRs in intermediate set.")

    # Handle deficit for final set (if sum < final_target_size)
    current_final_size = len(final_selected_pr_numbers)
    deficit_final = final_target_size - current_final_size
    if deficit_final > 0:
        logging.info(f"Stage 2 deficit: {deficit_final}. Sampling randomly from remaining intermediate pool.")
        # Remaining pool = intermediate indices - those already sampled in stage 2
        remaining_intermediate_indices = list(set(intermediate_clean_df.index) - sampled_indices_stage2)
        num_to_sample_deficit = min(deficit_final, len(remaining_intermediate_indices))
        if num_to_sample_deficit > 0:
            additional_indices = random.sample(remaining_intermediate_indices, num_to_sample_deficit)
            final_selected_pr_numbers.extend(additional_indices)
            sampled_indices_stage2.update(additional_indices)

    # Ensure uniqueness and trim/pad if needed to exactly match final_target_size
    final_selected_pr_numbers = list(set(final_selected_pr_numbers))
    if len(final_selected_pr_numbers) > final_target_size:
         logging.warning(f"Oversampled in Stage 2 ({len(final_selected_pr_numbers)} > {final_target_size}). Trimming.")
         final_selected_pr_numbers = random.sample(final_selected_pr_numbers, final_target_size)
    elif len(final_selected_pr_numbers) < final_target_size:
         logging.warning(f"Undersampled in Stage 2 ({len(final_selected_pr_numbers)} < {final_target_size}). Could not fill deficit completely.")
         # Optionally, you could pad by resampling, but usually better to accept smaller set


    # --- Final Selection & Verification ---
    selected_prs = [pr_map[num] for num in final_selected_pr_numbers if num in pr_map]
    logging.info(f"Two-stage sampling complete. Final selected: {len(selected_prs)} clean PRs.")

    if selected_prs:
        # Recalculate metrics for the final selected set for verification
        selected_clean_metrics_final = []
        for pr_num in final_selected_pr_numbers:
             metric_row = clean_df.loc[pr_num] # Get metrics from original full clean df
             selected_clean_metrics_final.append(metric_row.to_dict())
        selected_clean_df = pd.DataFrame(selected_clean_metrics_final)

        logging.info("--- Final Distribution Comparison ---")
        logging.info(f"Defect PRs ({len(defect_df)}):")
        logging.info(f"  Commit Count: Mean={defect_df['commit_count'].mean():.2f}, Var={defect_df['commit_count'].var():.2f}")
        logging.info(f"  Commit Length: Mean={defect_df['commit_length'].mean():.2f}, Var={defect_df['commit_length'].var():.2f}")
        logging.info(f"  Log Commit Length: Mean={defect_df['log_commit_length'].mean():.2f}, Var={defect_df['log_commit_length'].var():.2f}")
        logging.info(f"Sampled Clean PRs ({len(selected_clean_df)}):")
        logging.info(f"  Commit Count: Mean={selected_clean_df['commit_count'].mean():.2f}, Var={selected_clean_df['commit_count'].var():.2f}")
        logging.info(f"  Commit Length: Mean={selected_clean_df['commit_length'].mean():.2f}, Var={selected_clean_df['commit_length'].var():.2f}")
        logging.info(f"  Log Commit Length: Mean={selected_clean_df['log_commit_length'].mean():.2f}, Var={selected_clean_df['log_commit_length'].var():.2f}")
        logging.info("-----------------------------")

    return selected_prs



def collect_prs(repos):
    for repo_name, pr_count in repos:
        data_paths = [
            f"/SWRBench/data/{repo_name.replace('/', '__')}/collect_pr_run_0",
            f"/SWRBench/data/{repo_name.replace('/', '__')}/collect_pr_run_1",
            f"/SWRBench/data/{repo_name.replace('/', '__')}/collect_pr_run_2",
        ]
        for data_path in data_paths:
            os.makedirs(data_path, exist_ok=True)
            collect_repo_prs(repo_name, data_path)

def collect_train_prs():
    data_path = "/SWRBench/data_train/top_pypi.json"
    data = load_json(data_path)[:50]
    for item in data:
        repo_name = item["repo_name"]
        
        data_paths = [
            f"/SWRBench/data_train/{repo_name.replace('/', '__')}/collect_pr_run_0",
            f"/SWRBench/data_train/{repo_name.replace('/', '__')}/collect_pr_run_1",
            # f"/SWRBench/data_train/{repo_name.replace('/', '__')}/collect_pr_run_2",
        ]
        for data_path in data_paths:
            os.makedirs(data_path, exist_ok=True)
            collect_repo_prs(repo_name, data_path)


def collect_defect_prs(repos):
    all_sampled_prs = []
    for repo_name, pr_count in repos:
        data_paths = [
            f"/SWRBench/data/{repo_name.replace('/', '__')}/collect_pr_run_0",
            f"/SWRBench/data/{repo_name.replace('/', '__')}/collect_pr_run_1",
            f"/SWRBench/data/{repo_name.replace('/', '__')}/collect_pr_run_2",
        ]

        save_dir = os.path.dirname(data_paths[0])
        verified_results = verify_defect_prs(repo_name, data_paths, save_dir)
        analyse_prs([os.path.join(save_dir, 'verified_defect_results.jsonl')] + 
                    [os.path.join(data_path, 'analysis_results.jsonl') for data_path in data_paths], 
                    save_dir, save_suffix="_pre")
        filtered_prs = fliter_by_szz(save_dir, verified_results)
        save_jsonl(os.path.join(save_dir, 'verified_defect_results_szz_filtered.jsonl'), filtered_prs)
        filtered_prs = fliter_by_gitlog(save_dir, filtered_prs)
        filtered_prs = fliter_by_severity(save_dir, filtered_prs)
        save_jsonl(os.path.join(save_dir, 'verified_defect_results_severity_filtered.jsonl'), filtered_prs)
        sampled_prs = sample_prs(save_dir, filtered_prs)
        save_json(os.path.join(save_dir, 'verified_defect_results_sample.json'), sampled_prs)
        save_jsonl(os.path.join(save_dir, 'verified_defect_results_sample.jsonl'), sampled_prs)
        all_sampled_prs.extend(sampled_prs)
        analyse_prs([os.path.join(save_dir, 'verified_defect_results_sample.jsonl')], save_dir, save_suffix="_post")
    
    logging.info(f"样本选择完成，选择 {len(all_sampled_prs)} 个PR")
    save_jsonl("/SWRBench/data/verified_defect_results_all_sampled.jsonl", all_sampled_prs)
    save_json("/SWRBench/data/verified_defect_results_all_sampled.json", all_sampled_prs)

def collect_train_defect_prs():
    data_path = "/SWRBench/data_train/top_pypi.json"
    data = load_json(data_path)[:50]
    all_sampled_prs = []
    for item in data:
        repo_name = item["repo_name"]
        data_paths = [
            f"/SWRBench/data_train/{repo_name.replace('/', '__')}/collect_pr_run_0",
            f"/SWRBench/data_train/{repo_name.replace('/', '__')}/collect_pr_run_1",
            # f"/SWRBench/data_train/{repo_name.replace('/', '__')}/collect_pr_run_2",
        ]

        save_dir = os.path.dirname(data_paths[0])
        verified_results = verify_defect_prs(repo_name, data_paths, save_dir)
        analyse_prs([os.path.join(save_dir, 'verified_defect_results.jsonl')] + 
                    [os.path.join(data_path, 'analysis_results.jsonl') for data_path in data_paths], 
                    save_dir, save_suffix="_pre")
        # filtered_prs = fliter_by_szz(save_dir, verified_results)
        filtered_prs = verified_results
        save_jsonl(os.path.join(save_dir, 'verified_defect_results_szz_filtered.jsonl'), filtered_prs)
        filtered_prs = fliter_by_gitlog(save_dir, filtered_prs)
        filtered_prs = fliter_by_severity(save_dir, filtered_prs)
        save_jsonl(os.path.join(save_dir, 'verified_defect_results_severity_filtered.jsonl'), filtered_prs)
        # sampled_prs = sample_prs(save_dir, filtered_prs)
        sampled_prs = filtered_prs
        save_json(os.path.join(save_dir, 'verified_defect_results_sample.json'), sampled_prs)
        save_jsonl(os.path.join(save_dir, 'verified_defect_results_sample.jsonl'), sampled_prs)
        all_sampled_prs.extend(sampled_prs)
        analyse_prs([os.path.join(save_dir, 'verified_defect_results_sample.jsonl')], save_dir, save_suffix="_post")
    
    logging.info(f"样本选择完成，选择 {len(all_sampled_prs)} 个PR")
    save_jsonl("/SWRBench/data_train/verified_defect_results_all_sampled.jsonl", all_sampled_prs)
    save_json("/SWRBench/data_train/verified_defect_results_all_sampled.json", all_sampled_prs)

def collect_train_clean_prs():
    data_path = "/SWRBench/data_train/top_pypi.json"
    data = load_json(data_path)[:50]
    all_sampled_prs = []
    for item in data:
        repo_name = item["repo_name"]
        data_paths = [
            f"/SWRBench/data_train/{repo_name.replace('/', '__')}/collect_pr_run_0",
            f"/SWRBench/data_train/{repo_name.replace('/', '__')}/collect_pr_run_1",
            # f"/SWRBench/data_train/{repo_name.replace('/', '__')}/collect_pr_run_2",
        ]

        save_dir = os.path.dirname(data_paths[0])
        verified_results = verify_clean_prs(repo_name, data_paths, save_dir)
        # filtered_prs = fliter_by_szz(save_dir, verified_results)
        filtered_prs = verified_results
        verified_defect_results = load_jsonl(os.path.join(save_dir, 'verified_defect_results_sample.jsonl'))
        # filtered_prs = fliter_by_gitlog(save_dir, filtered_prs)
        sampled_prs = sample_clean_prs(save_dir, verified_defect_results, filtered_prs)
        all_sampled_prs.extend(sampled_prs)
    
    logging.info(f"样本选择完成，选择 {len(all_sampled_prs)} 个PR")
    save_json("/SWRBench/data_train/verified_clean_results_all_sampled.json", all_sampled_prs)
    save_jsonl("/SWRBench/data_train/verified_clean_results_all_sampled.jsonl", all_sampled_prs)

def collect_clean_prs(repos):
    all_sampled_prs = []
    for repo_name, pr_count in repos:
        data_paths = [
            f"/SWRBench/data/{repo_name.replace('/', '__')}/collect_pr_run_0",
            f"/SWRBench/data/{repo_name.replace('/', '__')}/collect_pr_run_1",
            f"/SWRBench/data/{repo_name.replace('/', '__')}/collect_pr_run_2",
        ]

        save_dir = os.path.dirname(data_paths[0])
        verified_results = verify_clean_prs(repo_name, data_paths, save_dir)
        filtered_prs = fliter_by_szz(save_dir, verified_results)
        verified_defect_results = load_jsonl(os.path.join(save_dir, 'verified_defect_results_sample.jsonl'))
        filtered_prs = fliter_by_gitlog(save_dir, filtered_prs)
        sampled_prs = sample_clean_prs(save_dir, verified_defect_results, filtered_prs)
        all_sampled_prs.extend(sampled_prs)
    
    logging.info(f"样本选择完成，选择 {len(all_sampled_prs)} 个PR")
    save_json("/SWRBench/data/verified_clean_results_all_sampled.json", all_sampled_prs)
    save_jsonl("/SWRBench/data/verified_clean_results_all_sampled.jsonl", all_sampled_prs)


def collect_final_prs():
    random.seed(42)
    # defect_prs = load_jsonl("/SWRBench/data/verified_defect_results_all_sampled.jsonl")
    defect_prs = load_jsonl("/SWRBench/data/final_correct_changed_data-0520.jsonl")
    clean_prs = load_jsonl("/SWRBench/data/verified_clean_results_all_sampled.jsonl")

    clean_prs = [pr_info for pr_info in clean_prs if f"{pr_info['repo'].replace('/', '__')}-{pr_info['pr_number']}" not in ['matplotlib__matplotlib-2275', 'psf__requests-562', 'scikit-learn__scikit-learn-568', 'sympy__sympy-196', 'sympy__sympy-2809']]
    
    final_prs = []
    defect_prs = random.sample(defect_prs, 500)
    save_json("/SWRBench/data/verified_final_defect_results.json", defect_prs)
    save_jsonl("/SWRBench/data/verified_final_defect_results.jsonl", defect_prs)
    defect_prs.sort(key=lambda x: (x['repo'], int(x['pr_number'])))
    
    clean_prs = random.sample(clean_prs, 500)
    save_json("/SWRBench/data/verified_final_clean_results.json", clean_prs)
    save_jsonl("/SWRBench/data/verified_final_clean_results.jsonl", clean_prs)
    clean_prs.sort(key=lambda x: (x['repo'], int(x['pr_number'])))
    
    for pr_info in defect_prs:
        final_prs.append(pr_info)
    for pr_info in clean_prs:
        final_prs.append(pr_info)

    save_json("/SWRBench/data/verified_final_results.json", final_prs)
    save_jsonl("/SWRBench/data/verified_final_results.jsonl", final_prs)


def collect_final_prs_v2():
    random.seed(42)
    defect_prs = load_jsonl("/SWRBench/data/final_correct_changed_data-0520.jsonl")
    clean_prs = load_jsonl("/SWRBench/data/verified_clean_results_all_sampled.jsonl")
    privos_prs = load_jsonl("/SWRBench/data/swr_datasets_0520_d3c3.jsonl")
    privos_prs_ids = [(pr_info['instance_id']) for pr_info in privos_prs]
    
    clean_prs = [pr_info for pr_info in clean_prs if f"{pr_info['repo'].replace('/', '__')}-{pr_info['pr_number']}" not in ['matplotlib__matplotlib-2275', 'psf__requests-562', 'scikit-learn__scikit-learn-568', 'sympy__sympy-196', 'sympy__sympy-2809']]
    privos_clean_prs = [pr_info for pr_info in clean_prs if f"{pr_info['repo'].replace('/', '__')}-{pr_info['pr_number']}" in privos_prs_ids]
    leaf_clean_prs = [pr_info for pr_info in clean_prs if f"{pr_info['repo'].replace('/', '__')}-{pr_info['pr_number']}" not in privos_prs_ids]
    
    final_prs = []
    defect_prs = random.sample(defect_prs, 500)
    defect_prs.sort(key=lambda x: (x['repo'], int(x['pr_number'])))
    
    clean_prs = random.sample(leaf_clean_prs, 200)
    clean_prs = clean_prs + privos_clean_prs
    clean_prs.sort(key=lambda x: (x['repo'], int(x['pr_number'])))
    
    for pr_info in defect_prs:
        final_prs.append(pr_info)
    for pr_info in clean_prs:
        final_prs.append(pr_info)

    save_json("/SWRBench/data/verified_final_results.json", final_prs)
    save_jsonl("/SWRBench/data/verified_final_results.jsonl", final_prs)


def collect_labelv2():
    import pdb; pdb.set_trace()
    random.seed(42)
    defect_prs = load_jsonl("/SWRBench/data/verified_defect_results_all_sampled.jsonl")
    labeled_prs = load_json("/SWRBench/data/verified_final_defect_results_0507.json")
    labeled_prs_ids = [(pr_info['repo'], pr_info['pr_number']) for pr_info in labeled_prs]
    repos = [
        'astropy/astropy', 
        'django/django', 
        'matplotlib/matplotlib', 
        'mwaskom/seaborn', 
        'pallets/flask', 
        'psf/requests', 
        'pydata/xarray', 
        'pylint-dev/pylint', 
        'pytest-dev/pytest'
    ]
    defect_prs = [pr_info for pr_info in defect_prs if pr_info['repo'] in repos]
    defect_prs = [pr_info for pr_info in defect_prs if (pr_info['repo'], pr_info['pr_number']) not in labeled_prs_ids]
    
    privos_prs = load_jsonl("/SWRBench/data/verified_final_defect_results-labelv2.jsonl")
    privos_prs.sort(key=lambda x: (x['repo'], x['pr_number']))
    privos_prs = privos_prs[:30]
    privos_prs_last_id = (privos_prs[-1]['repo'], privos_prs[-1]['pr_number'])
    defect_prs.sort(key=lambda x: (x['repo'], x['pr_number']))
    first_id = 0
    for i, pr_info in enumerate(defect_prs):
        if pr_info['repo'] == privos_prs_last_id[0] and pr_info['pr_number'] > privos_prs_last_id[1]:
            first_id = i
            break
    defect_prs = defect_prs[first_id:]

    final_prs = []
    defect_prs = random.sample(defect_prs, 120)
    defect_prs = defect_prs + privos_prs
    save_json("/SWRBench/data/verified_final_defect_results-labelv3.json", defect_prs)
    save_jsonl("/SWRBench/data/verified_final_defect_results-labelv3.jsonl", defect_prs)


def collect_labelv4():
    # import pdb; pdb.set_trace()
    random.seed(42)
    defect_prs = load_jsonl("/SWRBench/data/verified_defect_results_all_sampled.jsonl")
    labeled_prs = load_json("/SWRBench/data/verified_final_defect_results_0507.json")
    labeled_prs_ids = [(pr_info['repo'], pr_info['pr_number']) for pr_info in labeled_prs]
    repos = [
        'scikit-learn/scikit-learn', 
        'sphinx-doc/sphinx', 
        'sympy/sympy', 
    ]
    defect_prs = [pr_info for pr_info in defect_prs if pr_info['repo'] in repos]
    defect_prs = [pr_info for pr_info in defect_prs if (pr_info['repo'], pr_info['pr_number']) not in labeled_prs_ids]
    
    defect_prs = random.sample(defect_prs, 150)
    save_json("/SWRBench/data/verified_final_defect_results-labelv4.json", defect_prs)
    save_jsonl("/SWRBench/data/verified_final_defect_results-labelv4.jsonl", defect_prs)

def convert_dataset(verified_pr_paths):
    verified_pr_list = []
    for verified_pr_path in verified_pr_paths:
        verified_pr_list.extend(load_jsonl(verified_pr_path))
    
    swr_datasets = []
    for pr_info in verified_pr_list:
        pr_repo = pr_info['repo']
        instance_id = f"{pr_repo.replace('/', '__')}-{pr_info['pr_number']}"
        # import pdb; pdb.set_trace()
        pr_description = [e for e in pr_info['timeline'] if e['type'] == 'description']
        pr_description = pr_description[0]['body']
        
        # if len(pr_info['defects']) > 1:
        #     print(f"Multiple defects for {instance_id}")
        #     continue
        
        # import pdb; pdb.set_trace()
        all_commits = [e for e in pr_info['timeline'] if e['type'] == 'commit']
        instance_commits = []
        for commit in all_commits:
            if safe_parse_time(commit['date']) <= safe_parse_time(pr_info['created_at']):
                instance_commits.append(commit)
        instance_commits.sort(key=lambda x: safe_parse_time(x['date']))
        
        pr_defects = []
        for defect in pr_info['defects']:
            defect_commit = [commit for commit in all_commits if commit['sha'] == defect['change_introduction']['commit_sha']]
            defect_commit = defect_commit[0]
            pr_defects.append({
                "change_type": defect['change_type'],
                "change_introducing": defect['change_introduction'],
                "change_discussion": defect['discussion_evidence'],
                "change_resolve_info": defect['resolving_information'] if 'resolving_information' in defect else None,
            })
     
        
        swr_datasets.append({
            "repo": pr_repo,
            "instance_id": instance_id,
            "pr_title": pr_info['pr_title'],
            "pr_statement": pr_description,
            "change_introduced": True if len(pr_defects) > 0 else False,
            "base_commit": pr_info['base_commit'],
            "created_at": pr_info['created_at'],
            "changes": pr_defects,
            "pr_commits": instance_commits,
            "pr_timeline": pr_info['timeline'],
            "all_commits": all_commits,
        })
    print(f"Total number of swr datasets: {len(swr_datasets)}")
    save_jsonl("data/swr_datasets.jsonl", swr_datasets)



if __name__ == "__main__":
    repos = [
        ("astropy/astropy", 18000), # 372
        ("django/django", 20000), # 37
        ("matplotlib/matplotlib", 31000), # 377
        ("mwaskom/seaborn", 4000), # 24
        ("pallets/flask", 6000), # 15
        ("psf/requests", 7000), # 65
        ("pydata/xarray", 11000), # 232
        ("pylint-dev/pylint", 11000), # 166
        ("pytest-dev/pytest", 14000), # 113
        ("scikit-learn/scikit-learn", 32000), # 648
        ("sphinx-doc/sphinx", 14000), # 82
        ("sympy/sympy", 28000), # 659
    ]
    
    # collect_prs(repos)

    # collect_defect_prs(repos)
    
    collect_clean_prs(repos)
    
    # collect_labelv2()
    
    # collect_labelv4()
    
    
    # collect_final_prs()
    # collect_final_prs_v2()
    
    
    # verified_pr_paths = [
    #     # "/SWRBench/data/pallets__flask/verified_defect_results_sample.jsonl",
    #     "/SWRBench/data/verified_final_results.jsonl"
    # ]
    # convert_dataset(verified_pr_paths)
    
    # collect_train_prs()
    # collect_train_defect_prs()
    
    # collect_train_clean_prs()
    