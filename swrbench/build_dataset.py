import datetime
import json
import os
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from collections import defaultdict

from tqdm import tqdm

from utils import safe_parse_time


def load_jsonl(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f.readlines() if line.strip() != '']
    return data

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def save_jsonl(path, data):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

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
            defect_commit = [commit for commit in all_commits if commit['sha'] == defect['defect_introducing_commit']['sha']]
            defect_commit = defect_commit[0]
            
            pr_defects.append({
                "defect_type": defect['defect_type'],
                "defect_introducing_commit": defect['defect_introducing_commit'],
                "defect_discussion": defect['defect_discussion'],
                "defect_fix_commits": defect['fix_commits'],
            })
        
        swr_datasets.append({
            "repo": pr_repo,
            "instance_id": instance_id,
            "pr_title": pr_info['pr_title'],
            "pr_statement": pr_description,
            "defect_introduced": True if len(pr_defects) > 0 else False,
            "base_commit": pr_info['base_commit'],
            "created_at": pr_info['created_at'],
            "defects": pr_defects,
            "pr_commits": instance_commits,
            "pr_timeline": pr_info['timeline'],
            "all_commits": all_commits,
        })
    print(f"Total number of swr datasets: {len(swr_datasets)}")
    save_jsonl("data/swr_datasets.jsonl", swr_datasets)


def git_clone(repo_name, pr_number, target_path):
    cache_path = "data_train/.cache"
    cache_repo_path = f"{cache_path}/{repo_name.replace('/', '__')}"
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_path, exist_ok=True)
    
    # Clone repository to cache if it doesn't exist
    if not os.path.exists(cache_repo_path):
        try:
            subprocess.run(["git", "clone", f"https://github.com/{repo_name}.git", cache_repo_path], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository {repo_name}: {e.stderr.decode()}")
            return False
    
    # Fetch the specific PR ref into the cache repository
    try:
        fetch_ref = f"pull/{pr_number}/head"
        local_branch = f"pr_{pr_number}"
        # Check if the local branch already exists, skip if it does
        result = subprocess.run(["git", "branch"], cwd=cache_repo_path, check=True, capture_output=True, text=True)
        if not local_branch in result.stdout:
            # Fetch the PR head, overwriting the local branch ref if it exists (+ prefix)
            print(f"Fetching {fetch_ref} into {local_branch} for {repo_name} in {cache_repo_path}...")
            fetch_command = ["git", "fetch", "origin", f"+refs/{fetch_ref}:refs/heads/{local_branch}"]
            result = subprocess.run(fetch_command, cwd=cache_repo_path, check=True, capture_output=True, text=True)
            
    except subprocess.CalledProcessError as e:
        # Handle cases where the PR ref might not exist (e.g., PR closed, repo deleted, typo in number)
        print(f"Warning: Could not fetch {fetch_ref} for PR {pr_number} in {repo_name}. Error: {e.stderr}")
        return False 
    except Exception as e:
        print(f"Unexpected error fetching PR {pr_number} for {repo_name}: {str(e)}")
        return False

    # Create target directory if it doesn't exist
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Copy from cache to target path
    try:
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        shutil.copytree(cache_repo_path, target_path, symlinks=True)
        return True
    except Exception as e:
        print(f"Error copying repository to {target_path}: {str(e)}")
        return False
    

def build_instance(instance_id, pr_commits, instance_target_path):
    try:
        # Reset the working directory to a clean state before attempting checkouts
        try:
            reset_result = subprocess.run(["git", "reset", "--hard", "HEAD"],
                                          cwd=instance_target_path, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error resetting repository state for {instance_id}: {e.stderr}")
            return False

        # Create base branch pointing to the PARENT of the first PR commit
        first_pr_commit_sha = pr_commits[0]['sha']
        # Define the base commit as the parent of the first PR commit
        base_commit_parent = f"{first_pr_commit_sha}^"
        base_branch = f"base_branch"
        # Create the base branch pointing to the parent commit
        result = subprocess.run(["git", "checkout", "-b", base_branch, base_commit_parent],
                              cwd=instance_target_path, capture_output=True, text=True)
        # Check for errors immediately after creating the branch
        if result.returncode != 0:
            print(f"Error creating base branch ({base_branch}) from commit {base_commit_parent} for {instance_id}: {result.stderr}")
            return False

        # Now, proceed to check if the target defect commit exists or needs reconstruction
        # Check if defect commit exists using git cat-file -e (checks existence without outputting content)
        defect_sha = pr_commits[-1]['sha']
        try:
            result = subprocess.run(["git", "cat-file", "-e", defect_sha],
                                  cwd=instance_target_path, check=True, capture_output=True)
            commit_exists = True
        except subprocess.CalledProcessError:
            # Command returns non-zero if commit doesn't exist
            commit_exists = False

        defect_branch = "branch_under_review"
        if commit_exists:
            # If commit exists, create branch directly
            result = subprocess.run(["git", "checkout", "-b", defect_branch, defect_sha],
                                  cwd=instance_target_path, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error creating defect branch '{defect_branch}' for {instance_id}: {result.stderr}")
                return False
            
            # Verify the HEAD commit SHA after checkout
            try:
                head_sha_result = subprocess.run(["git", "rev-parse", "HEAD"],
                                                 cwd=instance_target_path, check=True, capture_output=True, text=True)
                current_head_sha = head_sha_result.stdout.strip()
                if current_head_sha != defect_sha:
                    print(f"Error: HEAD commit SHA mismatch for {instance_id} on branch '{defect_branch}'. Expected {defect_sha}, but got {current_head_sha}")
                    return False
            except subprocess.CalledProcessError as e:
                print(f"Error verifying HEAD commit SHA for {instance_id} on branch '{defect_branch}': {e.stderr}")
                return False
        else:
            print(f"Error: Target commit {defect_sha} does not exist for {instance_id}. Cannot create '{defect_branch}'.")
            return False
            # # If commit doesn't exist, create branch by applying commits
            # success = create_branch_from_commits(
            #     instance_target_path,
            #     base_branch,
            #     defect_branch,
            #     swe_instance['pr_commits']
            # )
            # if not success:
            #     return False
        
        print(f"Successfully set up branches for {instance_id}")
        return True
        
    except Exception as e:
        print(f"Unexpected error setting up git branches for {instance_id}: {str(e)}")
        return False


def build_dataset_project(verified_prs):
    not_success_results = []
    for pr_info in tqdm(verified_prs, desc="Building dataset"):
        instance_id = f"{pr_info['repo'].replace('/', '__')}-{pr_info['pr_number']}"
        if instance_id not in ["sphinx-doc__sphinx-2711"]:
            continue
        print(f"Current not success results: {not_success_results}")
        instance_target_path = f"data/projects/{pr_info['repo'].replace('/', '__')}/{instance_id}"
        pr_number = pr_info['pr_number']
        
        all_commits = [e for e in pr_info['timeline'] if e['type'] == 'commit']
        instance_commits = []
        for commit in all_commits:
            if safe_parse_time(commit['date']) <= safe_parse_time(pr_info['created_at']):
                instance_commits.append(commit)
        instance_commits.sort(key=lambda x: safe_parse_time(x['date']))

        success = git_clone(pr_info['repo'], pr_number, instance_target_path)
        if not success:
            print(f"Error cloning repository {pr_info['repo']} to {instance_target_path}")
            not_success_results.append(instance_id)
            continue
        
        success = build_instance(instance_id, instance_commits, instance_target_path)
        if not success:
            print(f"Error building instance {instance_id}")
            not_success_results.append(instance_id)
            continue
            
        # break  # For testing with first dataset only

    save_json("data/build_not_success_results.json", not_success_results)
    print(f"Total number of not success results: \n{not_success_results}")


# Helper function to process all PRs for a single repository
def _process_single_repo(repo_name, pr_list_for_repo, base_target_path_prefix):
    repo_not_success_results = []
    # Using leave=False for inner tqdm progress bar if you have an outer one for repos
    for pr_info in tqdm(pr_list_for_repo, desc=f"Processing PRs for {repo_name}", leave=False):
        instance_id = f"{repo_name.replace('/', '__')}-{pr_info['pr_number']}"
        instance_target_path = f"{base_target_path_prefix}/{repo_name.replace('/', '__')}/{instance_id}"

        # if os.path.exists(instance_target_path):
        #     # print(f"Skipping {instance_id}, path exists: {instance_target_path}") # Uncomment for debugging
        #     continue
        
        pr_number = pr_info['pr_number']
        
        all_commits = [e for e in pr_info['timeline'] if e['type'] == 'commit']
        instance_commits = []
        for commit in all_commits:
            if safe_parse_time(commit['date']) <= safe_parse_time(pr_info['created_at']):
                instance_commits.append(commit)
        instance_commits.sort(key=lambda x: safe_parse_time(x['date']))

        if not instance_commits:
            print(f"Warning: No instance commits found for {instance_id} (PR {pr_number} in repo {repo_name}) before PR creation date. Skipping.")
            repo_not_success_results.append(f"{instance_id}_no_commits_found")
            continue

        success_clone = git_clone(repo_name, pr_number, instance_target_path)
        if not success_clone:
            # git_clone already prints an error, so we just record the failure
            repo_not_success_results.append(instance_id)
            continue
        
        # Ensure instance_commits is not empty for build_instance
        # This check is technically redundant if the above check for instance_commits is comprehensive,
        # but kept for safety if logic changes. build_instance expects pr_commits[0].
        if not instance_commits: 
            print(f"Error: instance_commits is empty for {instance_id} before calling build_instance.")
            repo_not_success_results.append(instance_id + "_empty_commits_for_build")
            continue

        success_build = build_instance(instance_id, instance_commits, instance_target_path)
        if not success_build:
            # build_instance already prints an error
            repo_not_success_results.append(instance_id)
            continue
            
    return repo_not_success_results

def build_train_dataset_project(verified_prs):
    prs_by_repo = defaultdict(list)
    for pr_info in verified_prs:
        prs_by_repo[pr_info['repo']].append(pr_info)

    not_success_results = []
    base_target_path_prefix = "data_train/projects"
    
    # prs_by_repo = {"ray-project/ray": prs_by_repo["ray-project/ray"]}
    num_workers = len(prs_by_repo)
    if num_workers == 0:
        print("No repositories to process.")
        save_json("data_train/build_not_success_results.json", not_success_results)
        print(f"Total number of not success results: \n{not_success_results}")
        return

    # The print for "Current not success results" is removed from the loop as it's complex to manage with true parallelism this way.
    # We will collect all results and then save/print. Intermediate errors from processes are printed by them.

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_repo = {
            executor.submit(_process_single_repo, repo_name, pr_list, base_target_path_prefix): repo_name
            for repo_name, pr_list in prs_by_repo.items()
        }
        
        # tqdm for overall repository processing progress
        for future in tqdm(as_completed(future_to_repo), total=len(future_to_repo), desc="Building dataset (by repo)"):
            repo_name = future_to_repo[future]
            try:
                repo_failures = future.result()
                if repo_failures:
                    not_success_results.extend(repo_failures)
                    # This print gives feedback as each repo finishes if it had errors
                    print(f"Failures processed for repo {repo_name}: {repo_failures}") 
            except Exception as exc:
                print(f"Repository {repo_name} generated an exception during processing: {exc}")
                # Add a generic error for the repo, or mark all its PRs as failed if possible/desired
                not_success_results.append(f"REPO_PROCESSING_ERROR_{repo_name.replace('/', '__')}")

    save_json("data_train/build_not_success_results.json", not_success_results)
    print(f"Total number of not success results: \n{not_success_results}")


def check_sequence_commit_log(verified_prs):
    not_success_results = []
    for pr_info in tqdm(verified_prs, desc="Checking commit logs"): # Changed description
        instance_id = f"{pr_info['repo'].replace('/', '__')}-{pr_info['pr_number']}"
        instance_target_path = f"data/projects/{pr_info['repo'].replace('/', '__')}/{instance_id}"
        
        # Ensure the target path exists before proceeding
        if not os.path.exists(instance_target_path):
            print(f"Warning: Instance path {instance_target_path} not found for {instance_id}. Skipping check.")
            not_success_results.append(instance_id + "_path_not_found")
            continue

        # Extract instance commits similarly to build_dataset_project
        all_commits = [e for e in pr_info['timeline'] if e['type'] == 'commit']
        instance_commits = []
        for commit in all_commits:
            if safe_parse_time(commit['date']) <= safe_parse_time(pr_info['created_at']):
                instance_commits.append(commit)
        instance_commits.sort(key=lambda x: safe_parse_time(x['date']))
        
        # Extract the expected SHAs in order
        expected_shas = [commit['sha'] for commit in instance_commits]

        try:
            # Execute 'git log' in the specific repository path and get only SHAs
            # --format=%H gives only the full commit hash
            # --topo-order ensures commits are listed in topological order (matching typical PR commit sequences)
            # -n <count> limits the log to the number of expected commits to avoid comparing against unrelated history
            # Assumes the current HEAD of the checked-out branch_under_review should reflect the PR commits
            log_command = ["git", "log", f"-{len(expected_shas)}", "--format=%H", "--topo-order"]
            result = subprocess.run(log_command, cwd=instance_target_path, check=True, 
                                    capture_output=True, text=True)
            
            # Get the actual SHAs from the command output, splitting by newline and removing empty strings
            actual_shas = [sha for sha in result.stdout.strip().split('\n') if sha]
            
            # The git log might return commits in reverse chronological order (newest first)
            # while instance_commits is sorted chronologically (oldest first).
            # We need to reverse one of them for comparison. Let's reverse actual_shas.
            actual_shas.reverse()

            # Compare the actual SHAs with the expected SHAs
            if actual_shas == expected_shas:
                print(f"Commit sequence check passed for {instance_id}")
            else:
                print(f"Error: Commit sequence mismatch for {instance_id}")
                print(f"Expected SHAs (oldest to newest): {expected_shas}")
                print(f"Actual SHAs (oldest to newest):   {actual_shas}")
                not_success_results.append(instance_id)

        except subprocess.CalledProcessError as e:
            print(f"Error executing git log for {instance_id}: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error during commit check for {instance_id}: {str(e)}")

    # Save or print the results of the check
    if os.path.exists("data/check_log_not_success_results.json"):
        prior_not_success_results = load_json("data/check_log_not_success_results.json")
        not_success_results = list(set(prior_not_success_results + not_success_results))
    save_json("data/check_log_not_success_results.json", not_success_results)
    print(f"Finished checking commit logs. Issues found in: \n{not_success_results}")


def build_train_dataset_project_single(verified_prs):
    not_success_results = []
    for pr_info in tqdm(verified_prs, desc="Building dataset"):
        instance_id = f"{pr_info['repo'].replace('/', '__')}-{pr_info['pr_number']}"
        if instance_id not in ["pytorch__vision-851"]:
            continue
        print(f"Current not success results: {not_success_results}")
        instance_target_path = f"data/projects/{pr_info['repo'].replace('/', '__')}/{instance_id}"
        pr_number = pr_info['pr_number']
        
        all_commits = [e for e in pr_info['timeline'] if e['type'] == 'commit']
        instance_commits = []
        for commit in all_commits:
            if safe_parse_time(commit['date']) <= safe_parse_time(pr_info['created_at']):
                instance_commits.append(commit)
        instance_commits.sort(key=lambda x: safe_parse_time(x['date']))

        success = git_clone(pr_info['repo'], pr_number, instance_target_path)
        if not success:
            print(f"Error cloning repository {pr_info['repo']} to {instance_target_path}")
            not_success_results.append(instance_id)
            continue
        
        success = build_instance(instance_id, instance_commits, instance_target_path)
        if not success:
            print(f"Error building instance {instance_id}")
            not_success_results.append(instance_id)
            continue
            
        # break  # For testing with first dataset only

    save_json("data/build_not_success_results.json", not_success_results)
    print(f"Total number of not success results: \n{not_success_results}")

if __name__ == "__main__":
    defect_verified_prs = load_jsonl("/SWRBench/data/verified_defect_results_all_sampled.jsonl")
    clean_verified_prs = load_jsonl("/SWRBench/data/verified_clean_results_all_sampled.jsonl")
    # verified_prs = load_jsonl("/SWRBench/data/verified_results_all_sampled.jsonl")
    build_dataset_project(defect_verified_prs + clean_verified_prs)
    
    # check_sequence_commit_log(defect_verified_prs + clean_verified_prs)
    
    # defect_verified_prs = load_jsonl("/SWRBench/data_train/verified_defect_results_all_sampled_0521.jsonl")
    # clean_verified_prs = load_jsonl("/SWRBench/data_train/verified_clean_results_all_sampled_0521.jsonl")
    # # build_train_dataset_project(defect_verified_prs + clean_verified_prs)
    # build_train_dataset_project_single(defect_verified_prs + clean_verified_prs)
    
    