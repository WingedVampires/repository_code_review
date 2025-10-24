#!/usr/bin/env python3
import shutil
import subprocess
import json
import os
import argparse
from unidiff import PatchSet
from io import StringIO
from tqdm import tqdm

VENV_PATH = "/SWRBench/.venv"
SWR_AGENT_PATH = "/SWRBench/baselines/SWE-agent"


def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def save_jsonl(file_path, data):
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


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


def create_tasks(args):
    swr_datasets = load_jsonl(args.dataset_file)
    save_problem_statement_dir = os.path.join(args.output_dir, "problem_statement")
    
    prev_preds_path = os.path.join(args.output_dir, "generation.jsonl")
    prev_preds = {}
    if os.path.exists(prev_preds_path):
        prev_preds_list = load_jsonl(prev_preds_path)
        prev_preds = {item['instance_id']: item for item in prev_preds_list}
    os.makedirs(save_problem_statement_dir, exist_ok=True)
    swr_agent_datasets = []
    for swr_instance in swr_datasets:
        problem_statement = create_task_prompt(swr_instance)
        # if swr_instance['instance_id'] != 'astropy__astropy-77':
        #     continue
        swr_agent_datasets.append({
            "instance_id": swr_instance['instance_id'],
            "problem_statement": {"type": "text", "text": problem_statement, "id": swr_instance['instance_id']},
            "env": {
                "deployment": {
                    "type": "docker",
                    "image": "swe-agent-docker:python3.11",
                    "docker_args": ["-e", "all_proxy=http://172.17.0.1:10810", 
                                    "-v", "/SWRBench/.venv/lib/python3.11/site-packages/swerex:/root/python3.11/lib/python3.11/site-packages/swerex"],
                },
                "repo": {
                    "path": f"/SWRBench/data/projects/{swr_instance['repo'].replace('/', '__')}/{swr_instance['instance_id']}",
                },
            },
        })
        problem_statement_path = os.path.join(save_problem_statement_dir, f"{swr_instance['instance_id']}.md")
        with open(problem_statement_path, "w") as f:
            f.write(problem_statement)
        if swr_instance['instance_id'] in prev_preds and prev_preds[swr_instance['instance_id']]["review"] == "ERROR":
            shutil.rmtree(os.path.join(args.output_dir, "runs", swr_instance['instance_id']), ignore_errors=True)
    save_jsonl(os.path.join(args.output_dir, "swr_agent_datasets.jsonl"), swr_agent_datasets)
    return swr_agent_datasets


# 从任务结果中提取新文件内容
def extract_new_files(review_patch):
    if not review_patch:
        return None
    
    # 解析patch
    patch = PatchSet(StringIO(review_patch))
    
    # 收集所有新增的文件和内容
    new_files = {}
    for patched_file in patch:
        if patched_file.is_added_file:
            content = []
            for hunk in patched_file:
                for line in hunk:
                    if line.is_added:
                        content.append(line.value[1:] if line.value.startswith('+') else line.value)
            
            new_files[patched_file.path] = ''.join(content)
    
    return new_files


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
    parser.add_argument("--clean", action="store_true", default=False,
                        help="Clean the output directory")
    args = parser.parse_args()
    if args.clean:
        shutil.rmtree(os.path.join(args.output_dir, "runs"), ignore_errors=True)
    
    tasks = create_tasks(args)
    
    if args.model in ["deepseek/deepseek-chat"]:
        MODEL_KEY = f"DEEPSEEK_API_KEY={args.api_key}"
    else:
        MODEL_KEY = f"OPENAI_API_KEY={args.api_key} OPENAI_API_BASE={args.base_url}"
        
    cmd = f"""{MODEL_KEY} \
        sweagent run-batch \
            --config {os.path.join(SWR_AGENT_PATH, "config/code_review.yaml")} \
            --agent.model.name={args.model} \
            --agent.model.api_base={args.base_url} \
            --agent.model.temperature=1.0 \
            --agent.model.per_instance_call_limit=20 \
            --agent.model.max_input_tokens=65536 \
            --instances.type expert_file \
            --instances.path {os.path.join(args.output_dir, "swr_agent_datasets.jsonl")} \
            --instances.shuffle=False \
            --output_dir {os.path.join(args.output_dir, "runs")} \
            --num_workers {args.num_threads}
    """
    
    # Use the original command for Unix-like systems
    full_cmd = f"source {VENV_PATH}/bin/activate && {cmd}"
    process = None
    try:
        # Use preexec_fn to create a new process group
        process = subprocess.Popen(full_cmd, shell=True, executable="/bin/bash", 
                                  preexec_fn=os.setsid)
        process.wait()
        print(f"Completed review for {args.dataset_file}")
    except Exception as e:
        print(f"Error processing {args.dataset_file}: {str(e)}")
    finally:
        # Kill the entire process group, not just the shell
        if process and process.poll() is None:
            try:
                print("Terminating subprocess and all child processes...")
                # Send signal to the entire process group
                os.killpg(os.getpgid(process.pid), 15)  # SIGTERM to process group
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(os.getpgid(process.pid), 9)  # SIGKILL to process group
                except ProcessLookupError:
                    pass
                
        # Also find and kill any remaining 'sweagent run-batch' processes
        try:
            # Find all sweagent processes
            subprocess.run(
                "pkill -f 'sweagent run-batch'", 
                shell=True, 
                executable="/bin/bash"
            )
        except Exception:
            pass

    # Collect results
    results = []
    for task in tasks:
        result_path = os.path.join(args.output_dir, "runs", task['instance_id'], f"{task['instance_id']}.pred")
        review = "ERROR"
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                review_patch = json.load(f)
            review_patch = review_patch['model_patch']
            new_files = extract_new_files(review_patch)
            if new_files is not None and 'review_report.md' in new_files:
                review = new_files['review_report.md']
        results.append({
            "instance_id": task['instance_id'],
            "review": review
        })

    # Save results to output file
    with open(os.path.join(args.output_dir, "generation.jsonl"), "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"All tasks completed. Results saved to {args.output_dir}/generation.jsonl")

