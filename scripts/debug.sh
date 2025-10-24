#!/bin/bash

# set -x PYTHONPATH $PYTHONPATH (pwd)


# source /SWRBench/baselines/CodeReviewAgent/.venv/bin/activate && \
# OPENAI_API_KEY=your_openai_api_key OPENAI_API_BASE=your_openai_api_base_url \
# python3 /SWRBench/baselines/CodeReviewAgent/run.py --ifcode commit \
#     --config CodeReview \
#     --name swr-bench \
#     --model deepseek-chat \
#     --problem_description /SWRBench/logs/CR-Agent/deepseek-chat/runs/sympy__sympy-24194/problem_description.txt \
#     --log_dir /SWRBench/logs/CR-Agent/deepseek-chat/runs/sympy__sympy-24194

# source /SWRBench/baselines/CodeReviewAgent/.venv/bin/activate && \
# OPENAI_API_BASE=your_openai_api_base_url OPENAI_API_KEY=your_openai_api_key \
# python3 /SWRBench/baselines/CodeReviewAgent/run.py --ifcode commit \
#     --config CodeReview \
#     --name swr-bench \
#     --model deepseek-chat \
#     --problem_description /SWRBench/logs/CR-Agent/deepseek-chat/runs/astropy__astropy_10018/problem_description.txt \
#     --log_dir /SWRBench/logs/CR-Agent/deepseek-chat/runs/astropy__astropy_10018

# source .venv/bin/activate


source /SWRBench/baselines/pr-agent/.venv/bin/activate && python3 -m pr_agent.cli --pr_url '{"repo_path": "/SWRBench/data/projects/astropy__astropy/astropy__astropy-14682", "review_output_path": "/SWRBench/logs/PR-Agent/gemini-2.5-pro-exp-03-25/runs/astropy__astropy-14682", "base_branch_name": "base_branch", "target_branch_name": "branch_under_review"}' review
