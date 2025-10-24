import os
import json
import random
from tqdm import tqdm
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor
from github import Github
from collections import OrderedDict
from datetime import datetime, timezone
import itertools  # Add import for itertools

from utils import retry_function, save_jsonl, load_jsonl, safe_parse_time, load_json, save_json


GH_TOKENS = [
'your_github_token_1',
'your_github_token_2',
'your_github_token_3',
]


class RateLimiter:
    def __init__(self, calls_per_second=3):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0

    def __call__(self):
        now = time.time()
        elapsed = now - self.last_call
        wait = max(self.min_interval - elapsed, 0)
        if wait > 0:
            time.sleep(wait)
        self.last_call = time.time()

rate_limiter = RateLimiter()

@retry(stop=stop_after_attempt(10),
       wait=wait_exponential(multiplier=1, min=4, max=60),
       retry=retry_if_exception_type((requests.exceptions.RequestException, Exception)),
       before_sleep=lambda _: rate_limiter())
def retry_function(func, *args, **kwargs):
    return func(*args, **kwargs)



def clean_error_prs(repo_name):
    jsonl_file = f'data/train_prs_raw/{repo_name.replace("/", "__")}__prs.jsonl'
    if not os.path.exists(jsonl_file):
        return
    prs = load_jsonl(jsonl_file)
    new_prs = []
    for pr in prs:
        error_info = pr.get('ERROR_INFO', "")
        # if error_info.startswith("RetryError"):
        #     continue
        if error_info.startswith("ProxyError"):
            continue
        if error_info.startswith("ConnectionError"):
            continue
        if error_info.startswith("GithubException"):
            continue
        if error_info.startswith("AssertionError"):
            continue
        # if error_info.startswith("UnknownObjectException"):
        #     continue
        new_prs.append(pr)
    save_jsonl(jsonl_file, new_prs)
    
    
def show_error_prs(repo_name):
    jsonl_file = f'data/train_prs_raw/{repo_name.replace("/", "__")}__prs.jsonl'
    if not os.path.exists(jsonl_file):
        return
    prs = load_jsonl(jsonl_file)
    prs_error_type = {}
    for pr in prs:
        error_type = pr.get('ERROR_INFO', "SUCCESS")[:32]
        if error_type not in prs_error_type:
            prs_error_type[error_type] = 0
        prs_error_type[error_type] += 1
    print(f"{repo_name} has {len(prs)} PRs")
    for error_type, count in prs_error_type.items():
        print(f"{error_type}: {count}")


def get_pr_info(pr):
    pr_info = {
        'repo': pr.head.repo.full_name,
        'number': pr.number,
        'title': pr.title,
        'user': pr.user.login,
        'url': pr.html_url,
        'commits': pr.commits,
        'merge_commit_sha': pr.merge_commit_sha,
        'merged': pr.merged,
        'created_at': pr.created_at.replace(tzinfo=timezone.utc).isoformat(),
        'merged_at': pr.merged_at.replace(tzinfo=timezone.utc).isoformat() if pr.merged_at else None,
        'state': pr.state,
        'labels': [label.name for label in pr.labels],
        'issue_events': [event.event for event in pr.get_issue_events()],
        'base_commit': pr.base.sha,
        'timeline': []
    }

    timeline_dict = OrderedDict()

    if pr.body:
        key = f"description_{pr.created_at.isoformat()}"
        timeline_dict[key] = {
            'type': 'description',
            'user': pr.user.login,
            'body': pr.body,
            'created_at': pr.created_at.replace(tzinfo=timezone.utc).isoformat()
        }

    commits = retry_function(pr.get_commits)
    for commit in commits:
        key = f"commit_{commit.sha}"
        files = commit.files
        diff_text = "\n".join([f"{f.filename}\n{f.patch}" for f in files if f.patch])
        diff = [{'file': f.filename, 'patch': f.patch} for f in files if f.patch]
        timeline_dict[key] = {
            'type': 'commit',
            'sha': commit.sha,
            'message': commit.commit.message,
            'author': commit.commit.author.name,
            'author_email': commit.commit.author.email,
            'author_raw_date': commit.commit.author.date.isoformat(),
            'author_date': commit.commit.author.date.replace(tzinfo=timezone.utc).isoformat(),
            'committer': commit.commit.committer.name,
            'committer_email': commit.commit.committer.email,
            'raw_date': commit.commit.committer.date.isoformat(),
            'date': commit.commit.committer.date.replace(tzinfo=timezone.utc).isoformat(),
            'diff_text': diff_text,
            'diff': diff
        }

    comments = retry_function(pr.get_issue_comments)
    for comment in comments:
        key = f"comment_{comment.id}"
        timeline_dict[key] = {
            'type': 'comment',
            'id': comment.id,
            'user': comment.user.login,
            'body': comment.body,
            'created_at': comment.created_at.replace(tzinfo=timezone.utc).isoformat()
        }

    review_comments = retry_function(pr.get_review_comments)
    review_comments_dict = {}
    for comment in review_comments:
        comment_dict = {
            'type': 'review_comment',
            'id': comment.id,
            'in_reply_to_id': comment.in_reply_to_id,
            'user': comment.user.login,
            'body': comment.body,
            'created_at': comment.created_at.replace(tzinfo=timezone.utc).isoformat(),
            'path': comment.path,
            'diff_hunk': comment.diff_hunk,
            'start_line': comment.start_line,
            'original_start_line': comment.original_start_line,
            'start_side': comment.start_side,
            'line': comment.line,
            'original_line': comment.original_line,
            'side': comment.side,
            'original_position': comment.original_position,
            'position': comment.position,
            'subject_type': comment.subject_type,
            'reply': []
        }
        comment_dict['reply'].append({
            'id': comment_dict['id'],
            'user': comment_dict['user'],
            'body': comment_dict['body'],
            'created_at': comment_dict['created_at']
        })
        review_comments_dict[comment.id] = comment_dict
    
    for id, comment in review_comments_dict.items():
        if comment['in_reply_to_id'] is not None:
            root_reply_id = comment['in_reply_to_id']
            while review_comments_dict[root_reply_id]['in_reply_to_id'] is not None:
                root_reply_id = review_comments_dict[root_reply_id]['in_reply_to_id']
            review_comments_dict[root_reply_id]['reply'].append({
                'id': comment['id'],
                'user': comment['user'],
                'body': comment['body'],
                'created_at': comment['created_at']
            })
        else:
            timeline_dict[f"review_comment_{id}"] = comment

    reviews = retry_function(pr.get_reviews)
    for review in reviews:
        key = f"review_{review.id}"
        timeline_dict[key] = {
            'type': 'review',
            'id': review.id,
            'user': review.user.login,
            'body': review.body,
            'state': review.state,
            'created_at': review.submitted_at.replace(tzinfo=timezone.utc).isoformat() if review.submitted_at else None
        }

    pr_info['timeline'] = sorted(timeline_dict.values(), 
                                 key=lambda x: safe_parse_time(x.get('date') or x.get('created_at')) or datetime.min.replace(tzinfo=timezone.utc))
    for item in pr_info['timeline']:
        if item['type'] == 'review_comment':
            item['reply'] = sorted(item['reply'], 
                                   key=lambda x: safe_parse_time(x.get('date') or x.get('created_at')) or datetime.min.replace(tzinfo=timezone.utc))
    return pr_info

    
        
def get_all_prs(repo_name, specific_pr_ids_range):
    os.makedirs('data/train_prs_raw', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('data/train_prs_raw', 'collect_prs.log')),
            logging.StreamHandler()
        ]
    )
    
    jsonl_file = f'data/train_prs_raw/{repo_name.replace("/", "__")}__prs.jsonl'
    
    if os.path.exists(jsonl_file):
        all_pulls = load_jsonl(jsonl_file)
        all_pulls_ids = [pr['number'] for pr in all_pulls]
        logging.info(f"Loaded {len(all_pulls)} cached PRs")
    else:
        all_pulls = []
        all_pulls_ids = []
        logging.info(f"No cached PRs found")
        
    logging.info(f"Getting PRs of {repo_name}")
    specific_pr_ids = list(range(specific_pr_ids_range[0], specific_pr_ids_range[1] + 1))
    
    missing_pr_ids = [pr_id for pr_id in specific_pr_ids if pr_id not in all_pulls_ids]
    
    if not missing_pr_ids:
        logging.info("All specified PRs are already cached, no need to download")
        return all_pulls
    
    logging.info(f"Getting {len(missing_pr_ids)} PRs (total {len(specific_pr_ids)}, cached {len(all_pulls)} PRs)")
    
    file_lock = threading.Lock()
    
    gh_token_cycle = itertools.cycle(GH_TOKENS)
    def fetch_pr_worker(pr_id):
        try:
            gtoken = next(gh_token_cycle)
            repo = Github(gtoken, timeout=300).get_repo(repo_name)
            pr = repo.get_pull(pr_id)
            logging.info(f"获取到PR: {pr}")
            
            pr_info = retry_function(get_pr_info, pr)
            
            with file_lock:
                with open(jsonl_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(pr_info, ensure_ascii=False, default=str) + '\n')
            
        except Exception as e:
            with file_lock:
                with open(jsonl_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"number": pr_id, "ERROR": True, "ERROR_INFO": f"{e.__class__.__name__}: {str(e)}"}, ensure_ascii=False, default=str) + '\n')
                    
        return pr_id
    
    with ThreadPoolExecutor(max_workers=int(len(GH_TOKENS)*0.5)) as executor:
        results = list(tqdm(
            executor.map(fetch_pr_worker, missing_pr_ids),
            total=len(missing_pr_ids),
            desc="PRs"
        ))
    
    all_pulls = load_jsonl(jsonl_file)
    logging.info(f"Got {len(all_pulls)} PRs")
    
    return all_pulls

def get_eval_repo():
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
    
    for repo_name, pr_count in repos:
        # clean_error_prs(repo_name)
        # show_error_prs(repo_name)
        prs = get_all_prs(repo_name, specific_pr_ids_range=(0, pr_count))
        print(f"Got {len(prs)} PRs")


def get_train_repo():
    data_path = "/SWRBench/data/top_pypi.json"
    data = load_json(data_path)[45:50]
    for item in data:
        repo_name = item["repo_name"]
        pr_count = item["max_pr_count"]
        # show_error_prs(repo_name)
        # clean_error_prs(repo_name)
        # show_error_prs(repo_name)
        prs = get_all_prs(repo_name, specific_pr_ids_range=(0, pr_count))
        print(f"Got {len(prs)} PRs")


if __name__ == "__main__":
    # get_eval_repo()
    
    get_train_repo()