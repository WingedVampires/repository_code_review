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
    for commit in item["pr_commits"]:
        commit_code_change = f"Commit: {commit['sha']}\n"
        commit_code_change += f"Commit Message: {commit['message']}\n"
        commit_code_change += f"Commit Code Changes: \n"
        for diff in commit["diff"]:
            commit_code_change += f"{diff['file']}\n"
            commit_code_change += f"```\n{diff['patch']}\n```\n"
        pr_code_changes += f"{commit_code_change}\n"
    pr_code_changes = pr_code_changes.strip()
    
    return (
        "**Pull Request Details**: \n"
        f"- **Title**: {pr_title} \n"
        f"- **Description**: {pr_statement} \n"
        f"- **Code Changes**: {pr_code_changes} \n"
    )


def generate_task_base(args, item, logger, file_lock):
    # system_message = "You are a helpful assistant."
    system_message = None
    prinfo_prompt = create_prinfo_prompt(item)
    prompt = (
        "You are a senior project manager at a leading tech company, entrusted with maintaining the high standards of the codebase. A developer has submitted a pull request, and it’s your responsibility to review it thoroughly to ensure it meets the company’s quality and functionality requirements.\n"
        "Your task is to review the provided pull request, which includes the title, description, and code changes. You need to assess whether the pull request meets the standards for a good pull request or if there are any issues that need to be addressed.\n"
        "**Response Format**: In your response, please provide a brief summary of your assessment. Ensure your feedback is constructive and actionable. If you find no issues, please explicitly state that the pull request looks good and meets the standards.\n"
        "**Note**: The goal of this review is to maintain code quality and support the developer. Please provide respectful, specific, and helpful feedback to foster a collaborative environment. \n"
        f"<Start of Pull Request Details>\n"
        f"{prinfo_prompt}\n"
        f"<End of Pull Request Details>\n"
    )
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


def generate_task_refine(args, item, logger, file_lock):
    # system_message = "You are a helpful assistant."
    system_message = None
    prinfo_prompt = create_prinfo_prompt(item)
    prompt = (
        "You are a senior project manager at a leading tech company, entrusted with maintaining the high standards of the codebase. A developer has submitted a pull request, and it’s your responsibility to review it thoroughly to ensure it meets the company’s quality and functionality requirements.\n"
        "Your task is to review the provided pull request, which includes the title, description, and code changes. You need to assess whether the pull request meets the standards for a good pull request or if there are any issues that need to be addressed.\n"
        "**Response Format**: In your response, please provide a brief summary of your assessment. Ensure your feedback is constructive and actionable. If you find no issues, please explicitly state that the pull request looks good and meets the standards.\n"
        "**Note**: The goal of this review is to maintain code quality and support the developer. Please provide respectful, specific, and helpful feedback to foster a collaborative environment. \n"
        f"<Start of Pull Request Details>\n"
        f"{prinfo_prompt}\n"
        f"<End of Pull Request Details>\n"
    )
    # prompt = "Hi, how are you?"
    messages = create_messages(
        message=prompt,
        system_message=system_message
    )
    logger.info(f"Sending request for instance {item['instance_id']} ...")
    logger.debug(f"Prompt: \n{prompt}")
    # time.sleep(random.randint(5, 10))
    responses = []
    for i in range(3):
        response = run_chat(
            model=args.model, 
            messages=messages, 
            temperature=args.temperature, 
            max_tokens=args.max_tokens,
        )
        responses.append(response)
    if responses is None:
        logger.info(f"Failed to get response for instance {item['instance_id']}")
        return
    
    refine_prompt = (
        f"**Goal:** Generate a high-quality, consolidated code review report by analyzing the provided Pull Request (PR) details and critically evaluating three draft review reports.\n\n"
        f"**Role:** Act as an expert code reviewer tasked with synthesizing the most accurate, actionable, and constructive feedback possible.\n\n"
        f"**Process:**\n"
        f"1.  **Deeply Understand the Pull Request:** Carefully read and comprehend the `<Pull Request Details>` below. Identify the PR's objectives, the specific changes made (even if inferred from the description), and the context.\n"
        f"2.  **Critically Evaluate Draft Reports:** Analyze each of the three `<Draft Review Report>` sections provided. For *every* comment or point raised in these drafts, assess it based on the following criteria:\n"
        f"    *   **Accuracy:** Does the comment accurately reflect the code or changes described/implied in the `<Pull Request Details>`? **Discard any points that are factually incorrect or misinterpret the PR.**\n"
        f"    *   **Relevance & Significance:** Is the comment relevant to the changes in the PR? Does it address potential bugs, design flaws, security concerns, performance issues, deviations from standards, or areas for meaningful improvement? Prioritize significant points over minor nitpicks unless specifically requested.\n"
        f"    *   **Clarity & Actionability:** Is the comment clear, specific, and easy to understand? Does it suggest a concrete action the author can take? Vague or ambiguous comments should be refined for clarity or discarded if they cannot be made actionable.\n"
        f"    *   **Constructiveness:** Is the tone helpful and professional? Avoid points that are overly negative or unhelpful.\n"
        f"    *   **Redundancy:** Identify points that are essentially duplicates across the different drafts.\n"
        f"3.  **Synthesize the Final Consolidated Report:** Construct a single, coherent, and improved review report by performing the following:\n"
        f"    *   **Select Validated Points:** Only include points from the drafts that you have verified as accurate, relevant, and actionable based *strictly* on the `<Pull Request Details>`.\n"
        f"    *   **Consolidate & Refine:** Merge duplicate or very similar points into a single, well-phrased comment. Rephrase comments where necessary to improve clarity, conciseness, and maintain a constructive tone.\n"
        f"    *   **Structure Logically:** Organize the feedback in a clear and logical manner. Consider grouping comments by severity (e.g., Major Concerns, Suggestions, Minor Nitpicks), by file/module, or by theme. A brief introductory summary of the review might be beneficial.\n"
        f"    *   **Ensure Completeness (Based on Drafts):** Aim to cover the valid, important points raised across all three drafts, without introducing new points not derived from the drafts or the PR details.\n"
        f"    *   **Maintain Professional Tone:** The final report must be professional, objective, and focused on improving the code quality.\n\n"
        f"**Input Data:**\n\n"
        f"<Start of Pull Request Details>\n"
        f"{prinfo_prompt}\n"
        f"<End of Pull Request Details>\n\n"
        f"<Start of Draft Review Report 1>\n"
        f"{responses[0]}\n"
        f"<End of Draft Review Report 1>\n\n"
        f"<Start of Draft Review Report 2>\n"
        f"{responses[1]}\n"
        f"<End of Draft Review Report 2>\n\n"
        f"<Start of Draft Review Report 3>\n"
        f"{responses[2]}\n"
        f"<End of Draft Review Report 3>\n\n"
        f"**Task:** Now, generate the final, consolidated, and improved code review report based *only* on the provided PR details and the critically evaluated points from the draft reports, following all instructions above.\n"
        f"**Please directly output the final review report, without any other text.**\n"
    )
    
    messages = create_messages(
        message=refine_prompt,
        system_message=system_message
    )
    response = run_chat(
        model=args.model, 
        messages=messages, 
        temperature=args.temperature, 
        max_tokens=args.max_tokens
    )
    
    logger.debug(f"Received response for instance {item['instance_id']}: {response}")
    
    messages.append({"role": "assistant", "content": response})
    
    with file_lock:
        with open(args.output_file, "a") as f:
            f.write(json.dumps({
                "instance_id": item["instance_id"],
                "prompt": refine_prompt,
                "response": response,
                "review": response,
                "init_responses": responses
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
    
    
    
