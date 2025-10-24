#!/usr/bin/env python3
import shutil
import subprocess
import json
import sys
import os
import concurrent.futures
import argparse

from tqdm import tqdm
import tomli  # For Python < 3.11
# import tomllib  # For Python >= 3.11
import tomli_w  # For writing TOML files

VENV_PATH = "/SWRBench/baselines/pr-agent/.venv"

def load_toml_config(config_file):
    """
    Load configuration from a TOML file.
    
    Args:
        config_file (str): Path to the TOML configuration file
        
    Returns:
        dict: The configuration as a dictionary
    """
    try:
        with open(config_file, "rb") as f:
            return tomli.load(f)
    except Exception as e:
        print(f"Error loading config file {config_file}: {str(e)}")
        return {}

def save_toml_config(config_data, output_file):
    """
    Save configuration to a TOML file.
    
    Args:
        config_data (dict): Configuration data to save
        output_file (str): Path to save the TOML file
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, "wb") as f:
            tomli_w.dump(config_data, f)
        print(f"Config saved to {output_file}")
    except Exception as e:
        print(f"Error saving config to {output_file}: {str(e)}")

def run_pr_agent(repo_path, review_output_path, base_branch_name="base_branch", target_branch_name="branch_under_review"):
    """
    Run PR Agent with the specified parameters.
    
    Args:
        repo_path (str): Path to the repository
        review_output_path (str): Path to save review output
        base_branch_name (str): Base branch name
        target_branch_name (str): Target branch name
        config (dict, optional): Configuration containing API keys and other settings
    """
    pr_url = json.dumps({
        "repo_path": repo_path,
        "review_output_path": review_output_path,
        "base_branch_name": base_branch_name,
        "target_branch_name": target_branch_name
    })
    
    cmd = f"python3 -m pr_agent.cli --pr_url '{pr_url}' review"
    
    # Use the original command for Unix-like systems
    full_cmd = f"source {VENV_PATH}/bin/activate && {cmd}"
    try:
        print(f"Running command: {full_cmd}")
        subprocess.run(full_cmd, shell=True, check=True, executable="/bin/bash")
        # subprocess.run(full_cmd, shell=True, check=True, executable="/bin/bash", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Completed review for {repo_path}"
    except Exception as e:
        return f"Error processing {repo_path}: {str(e)}"

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

   
def create_tasks(args, previous_results):
    repo_path_prefix = "/SWRBench/data/projects/"
    swr_datasets = load_jsonl(args.dataset_file)
    swr_tasks = []
    for swr_instance in swr_datasets:
        if swr_instance['instance_id'] in previous_results:
            continue
        review_output_path = os.path.join(args.output_dir, "runs", swr_instance['instance_id'])
        shutil.rmtree(review_output_path, ignore_errors=True)
        task = {
            "instance_id": swr_instance['instance_id'],
            "repo_path": os.path.join(repo_path_prefix, swr_instance['repo'].replace("/", "__"), swr_instance['instance_id']),
            "review_output_path": review_output_path,
            "base_branch_name": "base_branch",
            "target_branch_name": "branch_under_review",
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
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature")
    parser.add_argument("--base-url", type=str, default="your_openai_api_base_url",
                        help="Base URL")
    parser.add_argument("--api-key", type=str, default="your_openai_api_key",
                        help="API key")
    parser.add_argument("--clean", action="store_true", help="Clean output directory")
    args = parser.parse_args()
    
    # Load configuration from TOML file if specified
    secret_config_path = "/SWRBench/baselines/pr-agent/pr_agent/settings/.secrets_template.toml"
    main_config_path = "/SWRBench/baselines/pr-agent/pr_agent/settings/configuration_template.toml"
    
    secret_config = load_toml_config(secret_config_path)
    if "deepseek" in args.model:
        secret_config['deepseek']['key'] = args.api_key
    elif "gemini" in args.model:
        secret_config['openai']['key'] = args.api_key
        secret_config['openai']['base_url'] = args.base_url
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    save_toml_config(secret_config, secret_config_path.replace("_template.toml", ".toml"))
    
    main_config = load_toml_config(main_config_path)
    main_config['config']['model'] = args.model
    main_config['config']['fallback_models'] = [args.model]
    main_config['config']['git_provider'] = "local"
    main_config['config']['publish_output'] = True
    main_config['config']['temperature'] = args.temperature
    save_toml_config(main_config, main_config_path.replace("_template.toml", ".toml"))
    # Load dataset
    previous_results = {}
    if args.clean:
        shutil.rmtree(os.path.join(args.output_dir, "runs"), ignore_errors=True)
    else:
        if os.path.exists(os.path.join(args.output_dir, "generation.jsonl")):
            previous_results = load_jsonl(os.path.join(args.output_dir, "generation.jsonl"))
            previous_results = {i["instance_id"]: i for i in previous_results if i["review"] != "ERROR" and i["review"] != "Preparing review..."}
    
    tasks = create_tasks(args, previous_results)
    
    print(f"Processing {len(tasks)} items with {args.num_threads} worker threads")
    # Process items in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # Create a list to store all the future objects
        futures = []
        
        # Submit all tasks to the executor
        for task in tasks:
            
            future = executor.submit(
                run_pr_agent, 
                task["repo_path"], 
                task["review_output_path"], 
                task["base_branch_name"], 
                task["target_branch_name"],
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
        system_prompt_path = os.path.join(task['review_output_path'], "system_prompt.txt")
        user_prompt_path = os.path.join(task['review_output_path'], "user_prompt.txt")
        review = "ERROR"
        system_prompt = "ERROR"
        user_prompt = "ERROR"
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                review = f.read()
        if os.path.exists(system_prompt_path):
            with open(system_prompt_path, "r") as f:
                system_prompt = f.read()
        if os.path.exists(user_prompt_path):
            with open(user_prompt_path, "r") as f:
                user_prompt = f.read()
        results.append({
            "instance_id": task['instance_id'],
            "repo_path": task['repo_path'],
            "review_output_path": task['review_output_path'],
            "review": review,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })

    for result in previous_results.values():
        results.append(result)

    # Save results to output file
    with open(os.path.join(args.output_dir, "generation.jsonl"), "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"All tasks completed. Results saved to {args.output_dir}/generation.jsonl")

