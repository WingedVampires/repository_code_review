#!/usr/bin/env python3
import shutil
import subprocess
import json
import sys
import os
import concurrent.futures
import argparse

from tqdm import tqdm

VENV_PATH = "/SWRBench/baselines/CodeReviewAgent/.venv"
CR_AGENT_PATH = "/SWRBench/baselines/CodeReviewAgent"

def run_cr_agent(task):
    """
    Run CR Agent with the specified parameters.
    
    Args:
        task (dict): Task dictionary containing all the information
    """
    
    cmd = f"""OPENAI_API_KEY={task['api_key']} OPENAI_API_BASE={task['base_url']} \
        python3 {CR_AGENT_PATH}/run.py --ifcode commit \
        --config {task['config']} \
        --name {task['name']} \
        --model {task['model']} \
        --problem_description {task['problem_description']} \
        --log_dir {task['review_output_path']}
    """
    
    # Use the original command for Unix-like systems
    full_cmd = f"source {VENV_PATH}/bin/activate && {cmd}"
    try:
        # subprocess.run(full_cmd, shell=True, check=True, executable="/bin/bash")
        subprocess.run(full_cmd, shell=True, check=True, executable="/bin/bash", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Completed review for {task['instance_id']}")
    except Exception as e:
        print(f"Error processing {task['instance_id']}: {str(e)}")

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def create_task_prompt(item):
    pr_title = item["pr_title"]
    pr_statement = item["pr_statement"]
    pr_code_changes = ""
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
        "You are a senior project manager at a leading tech company, entrusted with maintaining the high standards of the codebase. A developer has submitted a pull request, and it's your responsibility to review it thoroughly to ensure it meets the company's quality and functionality requirements.\n"
        "Your task is to review the provided pull request, which includes the title, description, and code changes. You need to assess whether the pull request meets the standards for a good pull request or if there are any issues that need to be addressed.\n"
        "**Note**: The goal of this review is to maintain code quality and support the developer. Please provide respectful, specific, and helpful feedback to foster a collaborative environment. \n"
        "**Pull Request Details**: \n"
        f"- **Title**: {pr_title} \n"
        f"- **Description**: {pr_statement} \n"
        f"- **Code Changes**: {pr_code_changes} \n"
    )
   
def create_tasks(args, previous_results):
    swr_datasets = load_jsonl(args.dataset_file)
    swr_tasks = []
    for swr_instance in swr_datasets:
        if swr_instance['instance_id'] in previous_results:
            continue
        task_prompt = create_task_prompt(swr_instance)
        task_output_path = os.path.join(args.output_dir, "runs", swr_instance['instance_id'])
        shutil.rmtree(task_output_path, ignore_errors=True)
        os.makedirs(task_output_path, exist_ok=True)
        problem_description_path = os.path.join(task_output_path, "problem_description.txt")
        with open(problem_description_path, "w") as f:
            f.write(task_prompt)
        task = {
            "instance_id": swr_instance['instance_id'],
            "config": "CodeReview",
            "name": "swr-bench",
            "model": args.model,
            "problem_description": problem_description_path,
            "review_output_path": task_output_path,
            "api_key": args.api_key,
            "base_url": args.base_url,
        }
        swr_tasks.append(task)

    return swr_tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PR Agent with multithreading")
    parser.add_argument("--num-threads", type=int, default=4, help="Maximum number of worker threads")
    parser.add_argument("--dataset-file", type=str, default="/SWRBench/data/swr_agent_datasets.jsonl",
                        help="Path to the dataset file")
    parser.add_argument("--output-dir", type=str, default="data/pr_agent_results",
                        help="Path to the output directory")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model name")
    parser.add_argument("--base-url", type=str, default="your_openai_api_base_url",
                        help="Base URL")
    parser.add_argument("--api-key", type=str, default="your_openai_api_key",
                        help="API key")
    parser.add_argument("--clean", action="store_true", help="Clean output directory")
    args = parser.parse_args()
    
    previous_results = {}
    if args.clean:
        shutil.rmtree(os.path.join(args.output_dir, "runs"), ignore_errors=True)
    else:
        if os.path.exists(os.path.join(args.output_dir, "generation.jsonl")):
            previous_results = load_jsonl(os.path.join(args.output_dir, "generation.jsonl"))
            previous_results = {i["instance_id"]: i for i in previous_results if i["review"] != "ERROR"}
    
    tasks = create_tasks(args, previous_results)
    
    print(f"Processing {len(tasks)} items with {args.num_threads} worker threads")
    # Process items in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # Create a list to store all the future objects
        futures = []
        
        # Submit all tasks to the executor
        for task in tasks:
            
            future = executor.submit(
                run_cr_agent, 
                task,
            )
            futures.append((future, task))
        
        # Process results as they complete
        for future, task in tqdm(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"Error in task for {task['instance_id']}: {str(e)}")

    # Collect results
    results = []
    for task in tasks:
        result_path = os.path.join(task['review_output_path'], "review.md")
        format_review_path = os.path.join(task['review_output_path'], "format_review.md")
        overall_review = "ERROR"
        format_review = "ERROR"
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                overall_review = f.read()
        if os.path.exists(format_review_path):
            with open(format_review_path, "r") as f:
                format_review = f.read()
        
        if overall_review == "ERROR" or format_review == "ERROR":
            review = "ERROR"
        else:
            review = f"Overall Review:\n{overall_review}\n\nFormat Review:\n{format_review}"
        results.append({
            "instance_id": task['instance_id'],
            "review": review,
        })
    
    for result in previous_results.values():
        results.append(result)

    # Save results to output file
    with open(os.path.join(args.output_dir, "generation.jsonl"), "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"All tasks completed. Results saved to {args.output_dir}/generation.jsonl")

