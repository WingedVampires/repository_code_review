import argparse
import logging
import os
from typing import List, Dict
import json
import time
import random

from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import run_chat
import pr_agent


log_file = "debug.log"
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s',  # Added filename and lineno
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

# Initialize logger for pr_agent module after basicConfig
pr_agent.LOGGER = logging.getLogger('pr_agent')
REPOS_DIR = "/SWRBench/data/projects"

def load_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def save_jsonl(file_path: str, data: List[Dict]):
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def load_dataset(args):
    
    dataset = load_jsonl(args.dataset_file)
    if args.instance_ids:
        dataset = [i for i in dataset if i["instance_id"] in args.instance_ids]
    if args.ignore_ids:
        dataset = [i for i in dataset if i["instance_id"] not in args.ignore_ids]
    
    task_dataset = []
    for item in dataset:
        task_dataset.append(item)

    return task_dataset

def create_messages(message, system_message=None):
    messages = [{"role": "system", "content": system_message}] if system_message is not None else []
    messages.append({"role": "user", "content": message})
    return messages

def create_prinfo_prompt(item):
    pr_title = item["pr_title"]
    pr_statement = item["pr_statement"]
    pr_code_changes = ""
    # import pdb; pdb.set_trace()
    file_paths = []
    for commit in item["pr_commits"]:
        commit_code_change = f"Commit: {commit['sha']}\n"
        commit_code_change += f"Commit Message: {commit['message']}\n"
        commit_code_change += f"Commit Code Changes: \n"
        for diff in commit["diff"]:
            commit_code_change += f"{diff['file']}\n"
            commit_code_change += f"```\n{diff['patch']}\n```\n"
            file_paths.append(diff['file'])
        pr_code_changes += f"{commit_code_change}\n"
    pr_code_changes = pr_code_changes.strip()
    
    base_branch_name = 'base_branch'
    target_branch_name = 'branch_under_review'
    repo_path = os.path.join(REPOS_DIR, item["repo"].replace("/", "__"), item["instance_id"])
    # As per original logic, TokenHandler is initialized with CHANGE_PROMPT template
    token_handler = pr_agent.TokenHandler(args.model, pr_agent.SYSTEM_PROMPT, pr_agent.USER_PROMPT)
    pr_code_changes, diff_files = pr_agent.get_pr_diff(repo_path, base_branch_name, target_branch_name, 
                            token_handler, args.model, add_line_numbers_to_hunks=True)
    
    file_paths = list(set(file_paths))
    static_analyzer_output = ""
    for file_path in file_paths:
        file_path = os.path.join(repo_path, file_path)
        if not file_path.endswith(".py") or not os.path.exists(file_path):
            continue
        
        output = os.popen(f"pylint --disable=C,W,E --score=no {file_path}").read()
        output_lines = output.split("\n")
        output_lines = [line for line in output_lines if "Unable to import" not in line]
        output = "\n".join(output_lines)
        static_analyzer_output += f"### {file_path} \n"
        static_analyzer_output += f"```\n{output}\n```\n"
        
        if len(static_analyzer_output) > 100000:
            break
        
    return (
        "### Comprehensive Code Review Report Generation\n\n"
        "You are an experienced software engineer tasked with providing a thorough code review. "
        "Your goal is to generate a comprehensive review report by analyzing both the output from a static code analyzer and the provided code changes independently, then synthesizing these into a single, actionable report.\n\n"
        "**Instructions:**\n"
        "1.  **Review Static Analyzer Output:** Carefully examine the `static_analyzer_output`. Identify the key issues it flags. Don't just list them; explain their potential impact if not immediately obvious, and consider if they are critical, major, or minor.\n"
        "2.  **Independent Code Analysis:** Independently review the `pr_code_changes`. Look for aspects such as (but not limited to):\n"
        "    *   Logic errors, potential bugs, or unhandled edge cases.\n"
        "    *   Readability, clarity, and maintainability of the code.\n"
        "    *   Adherence to programming best practices, design patterns, and established coding conventions.\n"
        "    *   Performance implications or inefficiencies.\n"
        "    *   Potential security vulnerabilities.\n"
        "    *   Adequacy and clarity of comments and documentation.\n"
        "    *   Test coverage (if discernible) and the quality/relevance of new or modified tests.\n"
        "    *   Opportunities for simplification, refactoring for better structure, or use of more idiomatic language constructs.\n"
        "3.  **Synthesize and Report:** Combine your findings from both steps into a single, coherent, and well-structured review report. \n"
        "    *   Clearly differentiate between issues raised by the static analyzer and your own observations. You can cite the static analyzer where its findings are relevant or use it as a starting point for deeper investigation.\n"
        "    *   If your independent analysis confirms or elaborates on a static analyzer finding, note that.\n"
        "    *   If your independent analysis finds issues not caught by the static analyzer, highlight these as new findings.\n"
        "    *   For each point, explain the issue, why it's a concern (impact), and provide specific, actionable suggestions for improvement. Reference specific line numbers or code snippets from `pr_code_changes` where appropriate.\n"
        "    *   Organize the report logically (e.g., by severity, by file, or by type of issue like 'Static Analyzer Findings', 'Logic Issues', 'Readability', etc.).\n"
        "    *   Conclude with an overall assessment if appropriate (e.g., 'Approved with minor suggestions', 'Requires major revisions').\n"
        "    *   Aim for a constructive, clear, and helpful tone.\n\n"
        f"### Static code analyzer output:\n```\n{static_analyzer_output}\n```\n\n"
        f"### Code difference (pr_code_changes):\n```\n{pr_code_changes}\n```\n\n"
        "### Comprehensive Code Review Report:\n"
    )



def generate_task_base(args, item, logger, file_lock):
    # system_message = "You are a helpful assistant."
    system_message = None
    prompt = create_prinfo_prompt(item)
    # prompt = "Hi, how are you?"
    messages = create_messages(
        message=prompt,
        system_message=system_message
    )
    logger.info(f"Sending request for instance {item['instance_id']} ...")
    logger.debug(f"Prompt: \n{prompt}")
    # time.sleep(random.randint(5, 10))
    response = run_chat(
        model=args.model, 
        messages=messages, 
        temperature=args.temperature, 
        max_tokens=args.max_tokens
    )
    if response is None:
        logger.info(f"Failed to get response for instance {item['instance_id']}")
        return
    
    logger.debug(f"Received response for instance {item['instance_id']}: {response}")
    
    messages.append({"role": "assistant", "content": response})
    
    with file_lock:
        with open(args.output_file, "a") as f:
            f.write(json.dumps({
                "instance_id": item["instance_id"],
                "prompt": prompt,
                "response": response,
                "review": response
            }) + "\n")


def generate(args):
    dataset = load_dataset(args)
    
    print("="*100)
    print(f"args: {args}")
    print("="*100)
    print(f"Evaluating {len(dataset)} instances")
    print(f"Dataset File: {args.dataset_file}")
    print(f"Model: {args.model}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Output File: {args.output_file}")
    print("="*100)
    
    print(f"Copying dataset {args.num_samples} times.")
    dataset = dataset * args.num_samples
    random.shuffle(dataset)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if os.path.exists(args.output_file) and not args.clean:
        previous_results = load_jsonl(args.output_file)
        previous_results = {i["instance_id"]: i for i in previous_results if i["review"] != "Error"}
        dataset = [i for i in dataset if i["instance_id"] not in previous_results]
        save_jsonl(args.output_file, list(previous_results.values()))
    else:
        with open(args.output_file, "w") as f:
            f.write("")
    
    log_file = args.output_file + ".log"
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    
    file_lock = threading.Lock()
    
    def process_item(item):
        # import pdb; pdb.set_trace()
        if args.refine:
            generate_task_refine(args, item, logger, file_lock)
        else:
            generate_task_base(args, item, logger, file_lock)
        return True

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(process_item, item) for item in dataset]
        for _ in tqdm(as_completed(futures), total=len(dataset), desc="Processing"):
            pass
    
    # for item in tqdm(dataset, total=len(dataset), desc="Processing"):
    #     process_item(item)
    
    print(f"Finished processing {len(dataset)} instances")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-file", type=str, help="Path to dataset file")
    parser.add_argument("--output-file", type=str, help="Path to output file")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--max-tokens", default=8192, type=int, help="Max tokens")
    parser.add_argument("--temperature", default=0.0, type=float, help="Temperature")
    parser.add_argument("--num-samples", default=1, type=int, help="Number of samples")
    parser.add_argument("--instance-ids", nargs="+", help="Instance ids")
    parser.add_argument("--ignore-ids", nargs="+", help="Ignore instance ids")
    parser.add_argument("--num-threads", default=1, type=int, help="Number of threads")
    parser.add_argument("--max-prompt-length", default=20000, type=int, help="Max prompt length")
    parser.add_argument("--clean", action="store_true", help="Clean output file")
    parser.add_argument("--refine", action="store_true", help="Refine review")
    args = parser.parse_args()
    
    generate(args)
    
    
    
