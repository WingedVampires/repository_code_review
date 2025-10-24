# set -e PYTHONPATH=$PYTHONPATH:$(pwd)
set -x PYTHONPATH $PYTHONPATH:(pwd)
export LITELLM_LOCAL_MODEL_COST_MAP=True


gemini-2.5-flash-preview-04-17,gemini-2.5-pro-exp-03-25,gemini-2.0-flash-exp,gemini-2.0-flash-thinking-exp

./scripts/run.sh gen hybrid_review gemini-2.5-flash-preview-04-17 Hybrid-Review/gemini-2.5-flash-preview
./scripts/run.sh gen hybrid_review gemini-2.5-pro-preview-03-25 Hybrid-Review/gemini-2.5-pro-preview
./scripts/run.sh gen hybrid_review claude-3-7-sonnet-20250219 Hybrid-Review/claude-3.7-sonnet
./scripts/run.sh gen hybrid_review deepseek-reasoner Hybrid-Review/deepseek-reasoner


./scripts/run.sh gen pr_agent o3-2025-04-16 PR-Agent/o3-2025-04-16
./scripts/run.sh gen pr_agent o4-mini-2025-04-16 PR-Agent/gpt-o4-mini
./scripts/run.sh gen pr_agent claude-sonnet-4-20250514 PR-Agent/claude-4-sonnet
./scripts/run.sh gen pr_agent claude-opus-4-20250514 PR-Agent/claude-4-opus
./scripts/run.sh gen pr_agent gemini-2.5-pro-preview-03-25 PR-Agent/gemini-2.5-pro-preview
./scripts/run.sh gen pr_agent claude-3-7-sonnet-20250219 PR-Agent/claude-3.7-sonnet
./scripts/run.sh gen pr_agent deepseek-reasoner PR-Agent/deepseek-reasoner
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview
./scripts/run.sh gen pr_agent gpt-4o-2024-11-20 PR-Agent/gpt-4o
./scripts/run.sh gen pr_agent deepseek-chat PR-Agent/deepseek-chat

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/o3-2025-04-16
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gpt-o4-mini
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/claude-4-sonnet
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/claude-4-opus
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-pro-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/claude-3.7-sonnet
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/deepseek-reasoner
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gpt-4o
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/deepseek-chat

./scripts/run.sh gen pr_agent vllm-qwen-7b PR-Agent/vllm-qwen-7b
./scripts/run.sh gen pr_agent vllm-qwen-14b PR-Agent/vllm-qwen-14b
./scripts/run.sh gen pr_agent vllm-qwen-32b PR-Agent/vllm-qwen-32b
./scripts/run.sh gen pr_agent vllm-qwen-r1-7b PR-Agent/vllm-qwen-r1-7b
./scripts/run.sh gen pr_agent vllm-qwen-r1-14b PR-Agent/vllm-qwen-r1-14b
./scripts/run.sh gen pr_agent vllm-qwen-r1-32b PR-Agent/vllm-qwen-r1-32b



./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-0.2-v1
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-0.2-v2
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-0.2-v3
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-0.2-v4
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-0.2-v5

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-0.2-v1
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-0.2-v2
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-0.2-v3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-0.2-v4
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-0.2-v5
# 1.57$ for 1000 cases


./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Hybrid-Review/gemini-2.5-flash-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Hybrid-Review/gemini-2.5-pro-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Hybrid-Review/claude-3.7-sonnet
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Hybrid-Review/deepseek-reasoner


./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/o3-2025-04-16
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gpt-o4-mini
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/claude-4-sonnet
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/claude-4-opus
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v11


./scripts/run.sh gen pr_agent gemini-2.5-pro-preview-03-25 PR-Agent/gemini-2.5-pro-preview
./scripts/run.sh gen pr_agent claude-3-7-sonnet-20250219 PR-Agent/claude-3.7-sonnet
./scripts/run.sh gen pr_agent deepseek-reasoner PR-Agent/deepseek-reasoner
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview
./scripts/run.sh gen pr_agent gpt-4o-2024-11-20 PR-Agent/gpt-4o
./scripts/run.sh gen pr_agent o3-mini-low PR-Agent/o3-mini-low

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/claude-3.7-sonnet
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-pro-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/deepseek-reasoner
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gpt-4o
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/o3-mini-low



./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v2
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v3
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v4
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v5



./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v2
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v4
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v5


./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-pro-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/claude-3.7-sonnet
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/deepseek-reasoner
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/deepseek-chat
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gpt-4o
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/o3-mini-high
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/o3-mini-medium
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/o3-mini-low
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-14b
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-32b
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-r1-7b
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-r1-14b
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-r1-32b

./scripts/run.sh gen pr_agent gemini-2.5-pro-preview-03-25 PR-Agent/gemini-2.5-pro-preview
./scripts/run.sh gen pr_agent gemini-2.5-pro-preview-03-25 PR-Agent/gemini-2.5-pro-preview-v2
./scripts/run.sh gen pr_agent gemini-2.5-pro-preview-03-25 PR-Agent/gemini-2.5-pro-preview-v3
./scripts/run.sh gen pr_agent gemini-2.5-pro-preview-03-25 PR-Agent/gemini-2.5-pro-preview-v4
./scripts/run.sh gen pr_agent gemini-2.5-pro-preview-03-25 PR-Agent/gemini-2.5-pro-preview-v5
./scripts/run.sh gen pr_agent gemini-2.5-pro-preview-03-25 PR-Agent/gemini-2.5-pro-preview-v6
./scripts/run.sh gen pr_agent gemini-2.5-pro-preview-03-25 PR-Agent/gemini-2.5-pro-preview-v7
./scripts/run.sh gen pr_agent gemini-2.5-pro-preview-03-25 PR-Agent/gemini-2.5-pro-preview-v8
./scripts/run.sh gen pr_agent gemini-2.5-pro-preview-03-25 PR-Agent/gemini-2.5-pro-preview-v9
./scripts/run.sh gen pr_agent gemini-2.5-pro-preview-03-25 PR-Agent/gemini-2.5-pro-preview-v10


./scripts/run.sh gen pr_agent vllm-qwen-7b-nc-50 PR-Agent/vllm-qwen-7b-nc-50
./scripts/run.sh gen pr_agent vllm-qwen-7b-nc-100 PR-Agent/vllm-qwen-7b-nc-100
./scripts/run.sh gen pr_agent vllm-qwen-7b-nc-150 PR-Agent/vllm-qwen-7b-nc-150
./scripts/run.sh gen pr_agent vllm-qwen-7b-nc-200 PR-Agent/vllm-qwen-7b-nc-200
./scripts/run.sh gen pr_agent vllm-qwen-7b-nc-250 PR-Agent/vllm-qwen-7b-nc-250
./scripts/run.sh gen pr_agent vllm-qwen-7b-nc-300 PR-Agent/vllm-qwen-7b-nc-300

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b-nc-50
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b-nc-100
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b-nc-150
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b-nc-200
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b-nc-250
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b-nc-300


./scripts/run.sh gen pr_agent vllm-qwen-7b-200 PR-Agent/vllm-qwen-7b-200
./scripts/run.sh gen pr_agent vllm-qwen-7b-250 PR-Agent/vllm-qwen-7b-250
./scripts/run.sh gen pr_agent vllm-qwen-7b-300 PR-Agent/vllm-qwen-7b-300
./scripts/run.sh gen pr_agent vllm-qwen-7b-350 PR-Agent/vllm-qwen-7b-350
./scripts/run.sh gen pr_agent vllm-qwen-7b-400 PR-Agent/vllm-qwen-7b-400
./scripts/run.sh gen pr_agent vllm-qwen-7b-450 PR-Agent/vllm-qwen-7b-450
./scripts/run.sh gen pr_agent vllm-qwen-7b-500 PR-Agent/vllm-qwen-7b-500
./scripts/run.sh gen pr_agent vllm-qwen-7b-580 PR-Agent/vllm-qwen-7b-580


./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b-200
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b-250
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b-300
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b-350
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b-400
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b-450
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b-500
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b-580


./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v2
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v3
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v4
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v5
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v6
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v7
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v8
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v9
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview-v10



./scripts/run.sh gen pr_agent deepseek-reasoner PR-Agent/deepseek-reasoner-v2
./scripts/run.sh gen pr_agent deepseek-reasoner PR-Agent/deepseek-reasoner-v3

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/gemini25flash-report-gemini25pro-v3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/gemini25pro-report-gemini25pro-v3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/gemini25flash-report-gemini25flash-v3


./scripts/run.sh gen npr_refine_review gemini-2.5-pro-preview-03-25 Refine-Review/gemini25pro-report-gemini25pro-v3 /SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview-v2/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview-v3/generation.jsonl
./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/gemini25flash-report-gemini25pro-v3 /SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview-v2/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview-v3/generation.jsonl

./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/gemini25flash-report-gemini25flash-v3 /SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview-v2/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview-v3/generation.jsonl


./scripts/run.sh gen pr_agent deepseek-chat PR-Agent/deepseek-chat


./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Base-Review/gemini-2.5-pro-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Base-Review/claude-3.7-sonnet
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Base-Review/deepseek-reasoner

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 CR-Agent/gemini-2.5-pro-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 CR-Agent/claude-3.7-sonnet
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 CR-Agent/deepseek-reasoner

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 SWR-Agent/gemini-2.5-pro-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 SWR-Agent/claude-3.7-sonnet

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-pro-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/claude-3.7-sonnet
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/deepseek-reasoner
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/deepseek-chat
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gpt-4o
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/o3-mini-high
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/o3-mini-medium
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/o3-mini-low
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-14b
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-32b
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-r1-7b
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-r1-14b
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-r1-32b

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/gemini25pro-report-gemini25pro
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/gemini25flash-report-3model
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/gemini25flash-report-gemini25flash
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/gemini25flash-report-gemini25pro
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-7b-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-7b-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-7b-report-10
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-14b-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-14b-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-14b-report-10
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-32b-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-32b-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-32b-report-10
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-7b-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-7b-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-7b-report-10
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-14b-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-14b-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-14b-report-10
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-32b-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-32b-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-32b-report-10


./scripts/run.sh gen pr_agent gpt-4o-2024-11-20 PR-Agent/gpt-4o
./scripts/run.sh gen pr_agent o3-mini-high PR-Agent/o3-mini-high
./scripts/run.sh gen pr_agent o3-mini-medium PR-Agent/o3-mini-medium
./scripts/run.sh gen pr_agent o3-mini-low PR-Agent/o3-mini-low


./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-pro-preview
./scripts/run.sh eval gemini-2.5-pro-preview-03-25 PR-Agent/gemini-2.5-pro-preview
./scripts/run.sh eval vllm-qwen-32b PR-Agent/gemini-2.5-pro-preview

export MULTIMODELREVIEWPATHS=/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/claude-3.7-sonnet/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/deepseek-reasoner/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/deepseek-chat/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gpt-4o/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gpt-o4-mini/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/o3-2025-04-16/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/claude-4-sonnet/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/claude-4-opus/generation.jsonl
./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/multimodelv2-report-10-v1 (echo $MULTIMODELREVIEWPATHS | cut -d, -f1-10)
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/multimodelv2-report-10-v1


export MULTIMODELREVIEWPATHS=/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/claude-3.7-sonnet/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/deepseek-reasoner/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/deepseek-chat/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gpt-4o/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gpt-o4-mini/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/o3-2025-04-16/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/claude-4-sonnet/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/claude-4-opus/generation.jsonl
./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/multimodelv2-report-1 (echo $MULTIMODELREVIEWPATHS | cut -d, -f1-1)
./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/multimodelv2-report-3 (echo $MULTIMODELREVIEWPATHS | cut -d, -f1-3)
./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/multimodelv2-report-5 (echo $MULTIMODELREVIEWPATHS | cut -d, -f1-5)
./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/multimodelv2-report-10 (echo $MULTIMODELREVIEWPATHS | cut -d, -f1-10)

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/multimodelv2-report-1
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/multimodelv2-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/multimodelv2-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/multimodelv2-report-10


export MULTIMODELREVIEWPATHS=/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/claude-3.7-sonnet/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/deepseek-reasoner/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/deepseek-chat/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gpt-4o/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gpt-o4-mini/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/o3-2025-04-16/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/claude-4-sonnet/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/claude-4-opus/generation.jsonl
./scripts/run.sh gen npr_refine_review gemini-2.5-pro-preview-03-25 Refine-Review/multimodelpro-report-1 (echo $MULTIMODELREVIEWPATHS | cut -d, -f1-1)
./scripts/run.sh gen npr_refine_review gemini-2.5-pro-preview-03-25 Refine-Review/multimodelpro-report-3 (echo $MULTIMODELREVIEWPATHS | cut -d, -f1-3)
./scripts/run.sh gen npr_refine_review gemini-2.5-pro-preview-03-25 Refine-Review/multimodelpro-report-5 (echo $MULTIMODELREVIEWPATHS | cut -d, -f1-5)
./scripts/run.sh gen npr_refine_review gemini-2.5-pro-preview-03-25 Refine-Review/multimodelpro-report-10 (echo $MULTIMODELREVIEWPATHS | cut -d, -f1-10)

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/multimodelpro-report-1
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/multimodelpro-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/multimodelpro-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/multimodelpro-report-10

export MULTIMODELREVIEWPATHS=/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/claude-3.7-sonnet/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/deepseek-reasoner/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/deepseek-chat/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gpt-4o/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/o3-mini-high/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/o3-mini-medium/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/o3-mini-low/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/o3-mini-high/generation.jsonl
./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/multimodel-report-1 (echo $MULTIMODELREVIEWPATHS | cut -d, -f1-1)
./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/multimodel-report-3 (echo $MULTIMODELREVIEWPATHS | cut -d, -f1-3)
./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/multimodel-report-5 (echo $MULTIMODELREVIEWPATHS | cut -d, -f1-5)
./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/multimodel-report-10 (echo $MULTIMODELREVIEWPATHS | cut -d, -f1-10)

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/multimodel-report-1
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/multimodel-report-1.1
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/multimodel-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/multimodel-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/multimodel-report-10

export GEMINI25PROREVIEWPATHS=/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview-v2/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview-v3/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview-v4/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview-v5/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview-v6/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview-v7/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview-v8/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview-v9/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview-v10/generation.jsonl
./scripts/run.sh gen npr_refine_review gemini-2.5-pro-preview-03-25 Refine-Review/gemini-2.5-pro-preview-report-1 (echo $GEMINI25PROREVIEWPATHS | cut -d, -f1-1)
./scripts/run.sh gen npr_refine_review gemini-2.5-pro-preview-03-25 Refine-Review/gemini-2.5-pro-preview-report-3 (echo $GEMINI25PROREVIEWPATHS | cut -d, -f1-3)
./scripts/run.sh gen npr_refine_review gemini-2.5-pro-preview-03-25 Refine-Review/gemini-2.5-pro-preview-report-5 (echo $GEMINI25PROREVIEWPATHS | cut -d, -f1-5)
./scripts/run.sh gen npr_refine_review gemini-2.5-pro-preview-03-25 Refine-Review/gemini-2.5-pro-preview-report-10 (echo $GEMINI25PROREVIEWPATHS | cut -d, -f1-10)

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/gemini-2.5-pro-preview-report-1
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/gemini-2.5-pro-preview-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/gemini-2.5-pro-preview-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/gemini-2.5-pro-preview-report-10



export GEMINI25FLASHREVIEWPATHS=/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview-v2/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview-v3/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview-v4/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview-v5/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview-v6/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview-v7/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview-v8/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview-v9/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview-v10/generation.jsonl
./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/gemini-2.5-flash-preview-report-1 (echo $GEMINI25FLASHREVIEWPATHS | cut -d, -f1-1)
./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/gemini-2.5-flash-preview-report-3 (echo $GEMINI25FLASHREVIEWPATHS | cut -d, -f1-3)
./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/gemini-2.5-flash-preview-report-5 (echo $GEMINI25FLASHREVIEWPATHS | cut -d, -f1-5)
./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/gemini-2.5-flash-preview-report-10 (echo $GEMINI25FLASHREVIEWPATHS | cut -d, -f1-10)

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/gemini-2.5-flash-preview-report-1
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/gemini-2.5-flash-preview-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/gemini-2.5-flash-preview-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/gemini-2.5-flash-preview-report-10


export VLLMQWEN7BREVIEWPATHS=/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-7b/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-7b-v2/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-7b-v3/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-7b-v4/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-7b-v5/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-7b-v6/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-7b-v7/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-7b-v8/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-7b-v9/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-7b-v10/generation.jsonl
./scripts/run.sh gen npr_refine_review vllm-qwen-7b Refine-Review/vllm-qwen-7b-report-1 (echo $VLLMQWEN7BREVIEWPATHS | cut -d, -f1-1)
./scripts/run.sh gen npr_refine_review vllm-qwen-7b Refine-Review/vllm-qwen-7b-report-3 (echo $VLLMQWEN7BREVIEWPATHS | cut -d, -f1-3)
./scripts/run.sh gen npr_refine_review vllm-qwen-7b Refine-Review/vllm-qwen-7b-report-5 (echo $VLLMQWEN7BREVIEWPATHS | cut -d, -f1-5)
./scripts/run.sh gen npr_refine_review vllm-qwen-7b Refine-Review/vllm-qwen-7b-report-10 (echo $VLLMQWEN7BREVIEWPATHS | cut -d, -f1-10)

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-7b-report-1
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-7b-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-7b-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-7b-report-10


export VLLMQWEN14BREVIEWPATHS=/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-14b/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-14b-v2/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-14b-v3/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-14b-v4/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-14b-v5/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-14b-v6/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-14b-v7/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-14b-v8/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-14b-v9/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-14b-v10/generation.jsonl
./scripts/run.sh gen npr_refine_review vllm-qwen-14b Refine-Review/vllm-qwen-14b-report-1 (echo $VLLMQWEN14BREVIEWPATHS | cut -d, -f1-1)
./scripts/run.sh gen npr_refine_review vllm-qwen-14b Refine-Review/vllm-qwen-14b-report-3 (echo $VLLMQWEN14BREVIEWPATHS | cut -d, -f1-3)
./scripts/run.sh gen npr_refine_review vllm-qwen-14b Refine-Review/vllm-qwen-14b-report-5 (echo $VLLMQWEN14BREVIEWPATHS | cut -d, -f1-5)
./scripts/run.sh gen npr_refine_review vllm-qwen-14b Refine-Review/vllm-qwen-14b-report-10 (echo $VLLMQWEN14BREVIEWPATHS | cut -d, -f1-10)

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-14b-report-1
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-14b-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-14b-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-14b-report-10

export VLLMQWEN32BREVIEWPATHS=/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-32b/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-32b-v2/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-32b-v3/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-32b-v4/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-32b-v5/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-32b-v6/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-32b-v7/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-32b-v8/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-32b-v9/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-32b-v10/generation.jsonl
./scripts/run.sh gen npr_refine_review vllm-qwen-32b Refine-Review/vllm-qwen-32b-report-1 (echo $VLLMQWEN32BREVIEWPATHS | cut -d, -f1-1)
./scripts/run.sh gen npr_refine_review vllm-qwen-32b Refine-Review/vllm-qwen-32b-report-3 (echo $VLLMQWEN32BREVIEWPATHS | cut -d, -f1-3)
./scripts/run.sh gen npr_refine_review vllm-qwen-32b Refine-Review/vllm-qwen-32b-report-5 (echo $VLLMQWEN32BREVIEWPATHS | cut -d, -f1-5)
./scripts/run.sh gen npr_refine_review vllm-qwen-32b Refine-Review/vllm-qwen-32b-report-10 (echo $VLLMQWEN32BREVIEWPATHS | cut -d, -f1-10)

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-32b-report-1
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-32b-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-32b-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-32b-report-10

export VLLMQWENR17BREVIEWPATHS=/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-7b/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-7b-v2/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-7b-v3/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-7b-v4/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-7b-v5/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-7b-v6/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-7b-v7/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-7b-v8/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-7b-v9/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-7b-v10/generation.jsonl
./scripts/run.sh gen npr_refine_review vllm-qwen-r1-7b Refine-Review/vllm-qwen-r1-7b-report-3 (echo $VLLMQWENR17BREVIEWPATHS | cut -d, -f1-3)
./scripts/run.sh gen npr_refine_review vllm-qwen-r1-7b Refine-Review/vllm-qwen-r1-7b-report-5 (echo $VLLMQWENR17BREVIEWPATHS | cut -d, -f1-5)
./scripts/run.sh gen npr_refine_review vllm-qwen-r1-7b Refine-Review/vllm-qwen-r1-7b-report-10 (echo $VLLMQWENR17BREVIEWPATHS | cut -d, -f1-10)
./scripts/run.sh gen npr_refine_review vllm-qwen-r1-7b Refine-Review/vllm-qwen-r1-7b-report-1 (echo $VLLMQWENR17BREVIEWPATHS | cut -d, -f1-1)

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-7b-report-1
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-7b-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-7b-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-7b-report-10


export VLLMQWENR114BREVIEWPATHS=/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-14b/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-14b-v2/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-14b-v3/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-14b-v4/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-14b-v5/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-14b-v6/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-14b-v7/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-14b-v8/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-14b-v9/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-14b-v10/generation.jsonl
./scripts/run.sh gen npr_refine_review vllm-qwen-r1-14b Refine-Review/vllm-qwen-r1-14b-report-3 (echo $VLLMQWENR114BREVIEWPATHS | cut -d, -f1-3)
./scripts/run.sh gen npr_refine_review vllm-qwen-r1-14b Refine-Review/vllm-qwen-r1-14b-report-5 (echo $VLLMQWENR114BREVIEWPATHS | cut -d, -f1-5)
./scripts/run.sh gen npr_refine_review vllm-qwen-r1-14b Refine-Review/vllm-qwen-r1-14b-report-10 (echo $VLLMQWENR114BREVIEWPATHS | cut -d, -f1-10)
./scripts/run.sh gen npr_refine_review vllm-qwen-r1-14b Refine-Review/vllm-qwen-r1-14b-report-1 (echo $VLLMQWENR114BREVIEWPATHS | cut -d, -f1-1)

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-14b-report-1
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-14b-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-14b-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-14b-report-10

export VLLMQWENR132BREVIEWPATHS=/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-32b/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-32b-v2/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-32b-v3/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-32b-v4/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-32b-v5/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-32b-v6/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-32b-v7/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-32b-v8/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-32b-v9/generation.jsonl,/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/vllm-qwen-r1-32b-v10/generation.jsonl
./scripts/run.sh gen npr_refine_review vllm-qwen-r1-32b Refine-Review/vllm-qwen-r1-32b-report-3 (echo $VLLMQWENR132BREVIEWPATHS | cut -d, -f1-3)
./scripts/run.sh gen npr_refine_review vllm-qwen-r1-32b Refine-Review/vllm-qwen-r1-32b-report-5 (echo $VLLMQWENR132BREVIEWPATHS | cut -d, -f1-5)
./scripts/run.sh gen npr_refine_review vllm-qwen-r1-32b Refine-Review/vllm-qwen-r1-32b-report-10 (echo $VLLMQWENR132BREVIEWPATHS | cut -d, -f1-10)
./scripts/run.sh gen npr_refine_review vllm-qwen-r1-32b Refine-Review/vllm-qwen-r1-32b-report-1 (echo $VLLMQWENR132BREVIEWPATHS | cut -d, -f1-1)

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-32b-report-1
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-32b-report-3
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-32b-report-5
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/vllm-qwen-r1-32b-report-10

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/vllm-qwen-7b-v2


./scripts/run.sh gen pr_agent vllm-qwen-7b PR-Agent/vllm-qwen-7b
./scripts/run.sh gen pr_agent vllm-qwen-7b PR-Agent/vllm-qwen-7b-v2
./scripts/run.sh gen pr_agent vllm-qwen-7b PR-Agent/vllm-qwen-7b-v3
./scripts/run.sh gen pr_agent vllm-qwen-7b PR-Agent/vllm-qwen-7b-v4
./scripts/run.sh gen pr_agent vllm-qwen-7b PR-Agent/vllm-qwen-7b-v5
./scripts/run.sh gen pr_agent vllm-qwen-7b PR-Agent/vllm-qwen-7b-v6
./scripts/run.sh gen pr_agent vllm-qwen-7b PR-Agent/vllm-qwen-7b-v7
./scripts/run.sh gen pr_agent vllm-qwen-7b PR-Agent/vllm-qwen-7b-v8
./scripts/run.sh gen pr_agent vllm-qwen-7b PR-Agent/vllm-qwen-7b-v9
./scripts/run.sh gen pr_agent vllm-qwen-7b PR-Agent/vllm-qwen-7b-v10

./scripts/run.sh gen pr_agent vllm-qwen-14b PR-Agent/vllm-qwen-14b
./scripts/run.sh gen pr_agent vllm-qwen-14b PR-Agent/vllm-qwen-14b-v2
./scripts/run.sh gen pr_agent vllm-qwen-14b PR-Agent/vllm-qwen-14b-v3
./scripts/run.sh gen pr_agent vllm-qwen-14b PR-Agent/vllm-qwen-14b-v4
./scripts/run.sh gen pr_agent vllm-qwen-14b PR-Agent/vllm-qwen-14b-v5
./scripts/run.sh gen pr_agent vllm-qwen-14b PR-Agent/vllm-qwen-14b-v6
./scripts/run.sh gen pr_agent vllm-qwen-14b PR-Agent/vllm-qwen-14b-v7
./scripts/run.sh gen pr_agent vllm-qwen-14b PR-Agent/vllm-qwen-14b-v8
./scripts/run.sh gen pr_agent vllm-qwen-14b PR-Agent/vllm-qwen-14b-v9
./scripts/run.sh gen pr_agent vllm-qwen-14b PR-Agent/vllm-qwen-14b-v10

./scripts/run.sh gen pr_agent vllm-qwen-32b PR-Agent/vllm-qwen-32b
./scripts/run.sh gen pr_agent vllm-qwen-32b PR-Agent/vllm-qwen-32b-v2
./scripts/run.sh gen pr_agent vllm-qwen-32b PR-Agent/vllm-qwen-32b-v3
./scripts/run.sh gen pr_agent vllm-qwen-32b PR-Agent/vllm-qwen-32b-v4
./scripts/run.sh gen pr_agent vllm-qwen-32b PR-Agent/vllm-qwen-32b-v5
./scripts/run.sh gen pr_agent vllm-qwen-32b PR-Agent/vllm-qwen-32b-v6
./scripts/run.sh gen pr_agent vllm-qwen-32b PR-Agent/vllm-qwen-32b-v7
./scripts/run.sh gen pr_agent vllm-qwen-32b PR-Agent/vllm-qwen-32b-v8
./scripts/run.sh gen pr_agent vllm-qwen-32b PR-Agent/vllm-qwen-32b-v9
./scripts/run.sh gen pr_agent vllm-qwen-32b PR-Agent/vllm-qwen-32b-v10

./scripts/run.sh gen pr_agent vllm-qwen-r1-7b PR-Agent/vllm-qwen-r1-7b
./scripts/run.sh gen pr_agent vllm-qwen-r1-7b PR-Agent/vllm-qwen-r1-7b-v2
./scripts/run.sh gen pr_agent vllm-qwen-r1-7b PR-Agent/vllm-qwen-r1-7b-v3
./scripts/run.sh gen pr_agent vllm-qwen-r1-7b PR-Agent/vllm-qwen-r1-7b-v4
./scripts/run.sh gen pr_agent vllm-qwen-r1-7b PR-Agent/vllm-qwen-r1-7b-v5
./scripts/run.sh gen pr_agent vllm-qwen-r1-7b PR-Agent/vllm-qwen-r1-7b-v6
./scripts/run.sh gen pr_agent vllm-qwen-r1-7b PR-Agent/vllm-qwen-r1-7b-v7
./scripts/run.sh gen pr_agent vllm-qwen-r1-7b PR-Agent/vllm-qwen-r1-7b-v8
./scripts/run.sh gen pr_agent vllm-qwen-r1-7b PR-Agent/vllm-qwen-r1-7b-v9
./scripts/run.sh gen pr_agent vllm-qwen-r1-7b PR-Agent/vllm-qwen-r1-7b-v10

./scripts/run.sh gen pr_agent vllm-qwen-r1-14b PR-Agent/vllm-qwen-r1-14b
./scripts/run.sh gen pr_agent vllm-qwen-r1-14b PR-Agent/vllm-qwen-r1-14b-v2
./scripts/run.sh gen pr_agent vllm-qwen-r1-14b PR-Agent/vllm-qwen-r1-14b-v3
./scripts/run.sh gen pr_agent vllm-qwen-r1-14b PR-Agent/vllm-qwen-r1-14b-v4
./scripts/run.sh gen pr_agent vllm-qwen-r1-14b PR-Agent/vllm-qwen-r1-14b-v5
./scripts/run.sh gen pr_agent vllm-qwen-r1-14b PR-Agent/vllm-qwen-r1-14b-v6
./scripts/run.sh gen pr_agent vllm-qwen-r1-14b PR-Agent/vllm-qwen-r1-14b-v7
./scripts/run.sh gen pr_agent vllm-qwen-r1-14b PR-Agent/vllm-qwen-r1-14b-v8
./scripts/run.sh gen pr_agent vllm-qwen-r1-14b PR-Agent/vllm-qwen-r1-14b-v9
./scripts/run.sh gen pr_agent vllm-qwen-r1-14b PR-Agent/vllm-qwen-r1-14b-v10

./scripts/run.sh gen pr_agent vllm-qwen-r1-32b PR-Agent/vllm-qwen-r1-32b
./scripts/run.sh gen pr_agent vllm-qwen-r1-32b PR-Agent/vllm-qwen-r1-32b-v2
./scripts/run.sh gen pr_agent vllm-qwen-r1-32b PR-Agent/vllm-qwen-r1-32b-v3
./scripts/run.sh gen pr_agent vllm-qwen-r1-32b PR-Agent/vllm-qwen-r1-32b-v4
./scripts/run.sh gen pr_agent vllm-qwen-r1-32b PR-Agent/vllm-qwen-r1-32b-v5
./scripts/run.sh gen pr_agent vllm-qwen-r1-32b PR-Agent/vllm-qwen-r1-32b-v6
./scripts/run.sh gen pr_agent vllm-qwen-r1-32b PR-Agent/vllm-qwen-r1-32b-v7
./scripts/run.sh gen pr_agent vllm-qwen-r1-32b PR-Agent/vllm-qwen-r1-32b-v8
./scripts/run.sh gen pr_agent vllm-qwen-r1-32b PR-Agent/vllm-qwen-r1-32b-v9
./scripts/run.sh gen pr_agent vllm-qwen-r1-32b PR-Agent/vllm-qwen-r1-32b-v10


./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Refine-Review/gemini25flash-report-gemini25flash

./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/gemini25flash-report-gemini25flash
./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/gemini25flash-report-gemini25pro
./scripts/run.sh gen npr_refine_review gemini-2.5-flash-preview-04-17 Refine-Review/gemini25flash-report-3model

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Base-Review/gemini-2.5-pro-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 CR-Agent/gemini-2.5-pro-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-pro-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 SWR-Agent/gemini-2.5-pro-preview

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Base-Review/deepseek-reasoner
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 CR-Agent/deepseek-reasoner
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/deepseek-reasoner
# ./scripts/run.sh eval gemini-2.5-flash-preview-04-17 SWR-Agent/deepseek-reasoner

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 Base-Review/claude-3.7-sonnet
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 CR-Agent/claude-3.7-sonnet
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/claude-3.7-sonnet
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 SWR-Agent/claude-3.7-sonnet

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview

./scripts/run.sh eval gemini-2.5-pro-preview-03-25 PR-Agent/gemini-2.5-pro-preview
./scripts/run.sh eval gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-pro-preview

./scripts/run.sh gen base_review gemini-2.5-pro-preview-03-25 Base-Review/gemini-2.5-pro-preview
./scripts/run.sh gen cr_agent gemini-2.5-pro-preview-03-25 CR-Agent/gemini-2.5-pro-preview
./scripts/run.sh gen pr_agent gemini-2.5-pro-preview-03-25 PR-Agent/gemini-2.5-pro-preview
./scripts/run.sh gen swr_agent openai/gemini-2.5-pro-preview-03-25 SWR-Agent/gemini-2.5-pro-preview

./scripts/run.sh gen base_review deepseek-reasoner Base-Review/deepseek-reasoner
./scripts/run.sh gen cr_agent deepseek-reasoner CR-Agent/deepseek-reasoner
./scripts/run.sh gen pr_agent deepseek-reasoner PR-Agent/deepseek-reasoner
# ./scripts/run.sh gen swr_agent deepseek-reasoner SWR-Agent/deepseek-reasoner

./scripts/run.sh gen base_review claude-3-7-sonnet-20250219 Base-Review/claude-3.7-sonnet
./scripts/run.sh gen cr_agent claude-3-7-sonnet-20250219 CR-Agent/claude-3.7-sonnet
./scripts/run.sh gen pr_agent claude-3-7-sonnet-20250219 PR-Agent/claude-3.7-sonnet
./scripts/run.sh gen swr_agent openai/claude-3-7-sonnet-20250219 SWR-Agent/claude-3.7-sonnet


./scripts/run.sh gen base_review gemini-2.5-flash-preview-04-17 Base-Review/gemini-2.5-flash-preview
./scripts/run.sh gen cr_agent gemini-2.5-flash-preview-04-17 CR-Agent/gemini-2.5-flash-preview
./scripts/run.sh gen pr_agent gemini-2.5-flash-preview-04-17 PR-Agent/gemini-2.5-flash-preview
./scripts/run.sh gen swr_agent openai/gemini-2.5-flash-preview-04-17 SWR-Agent/gemini-2.5-flash-preview


./scripts/run.sh gen base_review deepseek-chat Base-Review/deepseek-chat
./scripts/run.sh gen cr_agent deepseek-chat CR-Agent/deepseek-chat
./scripts/run.sh gen pr_agent deepseek-chat PR-Agent/deepseek-chat
./scripts/run.sh gen swr_agent deepseek-chat SWR-Agent/deepseek-chat


./scripts/run.sh gen base_review gpt-4o Base-Review/gpt-4o
./scripts/run.sh gen cr_agent gpt-4o CR-Agent/gpt-4o
./scripts/run.sh gen pr_agent gpt-4o PR-Agent/gpt-4o
./scripts/run.sh gen swr_agent gpt-4o SWR-Agent/gpt-4o


./scripts/run.sh eval gemini-2.5-pro-exp-03-25 Base-Review/gemini-2.5-pro-exp-03-25
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 CR-Agent/gemini-2.5-pro-exp-03-25
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/gemini-2.5-pro-exp-03-25
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 SWR-Agent/gemini-2.5-pro-exp-03-25
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 NPR-Refine-Review/gemini-2.5-pro-exp-03-25

./scripts/run.sh eval gemini-2.5-pro-exp-03-25 Base-Review/deepseek-reasoner
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 CR-Agent/deepseek-reasoner
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/deepseek-reasoner
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 NPR-Refine-Review/deepseek-reasoner

./scripts/run.sh eval gemini-2.5-pro-exp-03-25 Base-Review/gpt-4o-2024-11-20
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 CR-Agent/gpt-4o-2024-11-20
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/gpt-4o-2024-11-20
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 SWR-Agent/gpt-4o-2024-11-20
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 NPR-Refine-Review/gpt-4o-2024-11-20

./scripts/run.sh gen base_review deepseek-reasoner Base-Review/deepseek-reasoner
./scripts/run.sh gen cr_agent deepseek-reasoner CR-Agent/deepseek-reasoner
./scripts/run.sh gen pr_agent deepseek-reasoner PR-Agent/deepseek-reasoner
# ./scripts/run.sh gen swr_agent deepseek-chat SWR-Agent/deepseek-chat

./scripts/run.sh gen base_review gpt-4o-2024-11-20 Base-Review/gpt-4o-2024-11-20
./scripts/run.sh gen cr_agent gpt-4o-2024-11-20 CR-Agent/gpt-4o-2024-11-20
./scripts/run.sh gen pr_agent gpt-4o-2024-11-20 PR-Agent/gpt-4o-2024-11-20
./scripts/run.sh gen swr_agent openai/gpt-4o-2024-11-20 SWR-Agent/gpt-4o-2024-11-20
./scripts/run.sh gen npr_refine_review gpt-4o-2024-11-20 NPR-Refine-Review/gpt-4o-2024-11-20


./scripts/run.sh gen base_review gemini-2.5-pro-exp-03-25 Base-Review/gemini-2.5-pro-exp-03-25
./scripts/run.sh gen cr_agent gemini-2.5-pro-exp-03-25 CR-Agent/gemini-2.5-pro-exp-03-25
./scripts/run.sh gen pr_agent gemini-2.5-pro-exp-03-25 PR-Agent/gemini-2.5-pro-exp-03-25
./scripts/run.sh gen swr_agent openai/gpt-4o SWR-Agent/gemini-2.5-pro-exp-03-25

./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/gemini-2.5-pro-exp-03-25
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 Base-Review/gemini-2.5-pro-exp-03-25

./scripts/run.sh eval gemini-2.0-flash-exp NPR-Refine-Review/gemini-2.5-pro-exp-03-25
./scripts/run.sh eval deepseek-chat NPR-Refine-Review/gemini-2.5-pro-exp-03-25
./scripts/run.sh eval gpt-4o-mini NPR-Refine-Review/gemini-2.5-pro-exp-03-25
./scripts/run.sh eval gpt-4o NPR-Refine-Review/gemini-2.5-pro-exp-03-25
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 NPR-Refine-Review/gemini-2.5-pro-exp-03-25

./scripts/run.sh eval gemini-2.5-flash-preview-04-17 NPR-Refine-Review/gemini-2.5-pro-exp-03-25

./scripts/run.sh gen npr_refine_review gemini-2.5-pro-exp-03-25 NPR-Refine-Review/gemini-2.5-pro-exp-03-25
./scripts/run.sh gen npr_refine_review deepseek-reasoner NPR-Refine-Review/deepseek-reasoner

./scripts/run.sh gen refine_review gemini-2.5-pro-exp-03-25 Refine-Review/gemini-2.5-pro-exp-03-25
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 Refine-Review/gemini-2.5-pro-exp-03-25

./scripts/run.sh eval gemini-2.5-pro-exp-03-25 Base-Review/deepseek-chat
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 Base-Review/deepseek-reasoner
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 Base-Review/gemini-2.0-flash-exp
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 Base-Review/gemini-2.0-flash-thinking-exp
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 Base-Review/gemini-2.5-pro-exp-03-25


./scripts/run.sh eval gemini-2.5-pro-exp-03-25 CR-Agent/deepseek-chat
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 CR-Agent/deepseek-reasoner
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 CR-Agent/gemini-2.0-flash-exp
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 CR-Agent/gemini-2.0-flash-thinking-exp
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 CR-Agent/gemini-2.5-pro-exp-03-25


./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/deepseek-chat
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/deepseek-reasoner
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/gemini-2.0-flash-exp
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/gemini-2.0-flash-thinking-exp
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/gemini-2.5-pro-exp-03-25
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/gemini-2.5-pro-exp-03-25-0414
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/gemini-2.5-pro-exp-03-25-0415
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/gemini-2.5-pro-exp-03-25-041501
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/gemini-2.5-pro-exp-03-25-041502
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/gemini-2.5-pro-exp-03-25-041503tmp10

./scripts/run.sh eval gemini-2.5-pro-exp-03-25 SWR-Agent/deepseek-chat
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 SWR-Agent/gemini-2.0-flash-exp
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 SWR-Agent/gemini-2.5-pro-exp-03-25


./scripts/run.sh eval gemini-2.5-pro-exp-03-25 NPR-Review/gemini-2.0-flash-exp
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 NPR-Review/gemini-2.5-pro-exp-03-25-0415
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 NPR-Review/gemini-2.5-pro-exp-03-25-0415commitnodesc
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 NPR-Review/gemini-2.5-pro-exp-03-25-0415cutdiffnodesc
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 NPR-Review/gemini-2.5-pro-exp-03-25-0415desc
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 NPR-Review/gemini-2.5-pro-exp-03-25-0415mdout
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 NPR-Review/gemini-2.5-pro-exp-03-25-0415ori
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 NPR-Review/gemini-2.5-pro-exp-03-25-041501
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 NPR-Review/gemini-2.5-pro-exp-03-25-041502



./scripts/run.sh gen npr_review gemini-2.5-pro-exp-03-25 NPR-Review/gemini-2.5-pro-exp-03-25
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 NPR-Review/gemini-2.5-pro-exp-03-25


./scripts/run.sh gen base_review gemini-2.0-flash-thinking-exp Base-Review/gemini-2.0-flash-thinking-exp
./scripts/run.sh gen cr_agent gemini-2.0-flash-thinking-exp CR-Agent/gemini-2.0-flash-thinking-exp
./scripts/run.sh gen pr_agent openai/gemini-2.0-flash-thinking-exp PR-Agent/gemini-2.0-flash-thinking-exp
./scripts/run.sh gen swr_agent openai/gemini-2.0-flash-thinking-exp SWR-Agent/gemini-2.0-flash-thinking-exp


./scripts/run.sh eval gemini-2.5-pro-exp-03-25 Base-Review/gemini-2.0-flash-thinking-exp
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 CR-Agent/gemini-2.0-flash-thinking-exp

./scripts/run.sh eval gemini-2.5-pro-exp-03-25 CR-Agent/deepseek-reasoner
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/gemini-2.0-flash-thinking-exp
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/deepseek-reasoner

./scripts/run.sh eval gemini-2.5-pro-exp-03-25 SWR-Agent/gemini-2.0-flash-thinking-exp


./scripts/run.sh gen base_review deepseek-reasoner Base-Review/deepseek-reasoner
./scripts/run.sh gen cr_agent deepseek-reasoner CR-Agent/deepseek-reasoner
./scripts/run.sh gen pr_agent openai/deepseek-reasoner PR-Agent/deepseek-reasoner
./scripts/run.sh gen swr_agent openai/deepseek-reasoner SWR-Agent/deepseek-reasoner


./scripts/run.sh eval gemini-2.5-pro-exp-03-25 Base-Review/deepseek-reasoner
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 SWR-Agent/deepseek-reasoner

./scripts/run.sh gen base_review deepseek-chat Base-Review/deepseek-chat
./scripts/run.sh gen cr_agent deepseek-chat CR-Agent/deepseek-chat
./scripts/run.sh gen pr_agent openai/deepseek-chat PR-Agent/deepseek-chat
./scripts/run.sh gen swr_agent openai/deepseek-chat SWR-Agent/deepseek-chat

./scripts/run.sh eval gemini-2.5-pro-exp-03-25 Base-Review/deepseek-chat
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 CR-Agent/deepseek-chat
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/deepseek-chat
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 SWR-Agent/deepseek-chat

./scripts/run.sh gen base_review gemini-2.0-flash-exp Base-Review/gemini-2.0-flash-exp
./scripts/run.sh gen cr_agent gemini-2.0-flash-exp CR-Agent/gemini-2.0-flash-exp
./scripts/run.sh gen pr_agent openai/gemini-2.0-flash-exp PR-Agent/gemini-2.0-flash-exp
./scripts/run.sh gen swr_agent openai/gemini-2.0-flash-exp SWR-Agent/gemini-2.0-flash-exp

./scripts/run.sh eval gemini-2.5-pro-exp-03-25 Base-Review/gemini-2.0-flash-exp
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 CR-Agent/gemini-2.0-flash-exp
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/gemini-2.0-flash-exp
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 SWR-Agent/gemini-2.0-flash-exp

./scripts/run.sh gen base_review gemini-2.5-pro-exp-03-25 Base-Review/debug

./scripts/run.sh gen base_review gemini-2.5-pro-exp-03-25 Base-Review/gemini-2.5-pro-exp-03-25
./scripts/run.sh gen cr_agent gemini-2.5-pro-exp-03-25 CR-Agent/gemini-2.5-pro-exp-03-25
./scripts/run.sh gen pr_agent openai/gemini-2.5-pro-exp-03-25 PR-Agent/gemini-2.5-pro-exp-03-25
./scripts/run.sh gen swr_agent openai/gemini-2.5-pro-exp-03-25 SWR-Agent/gemini-2.5-pro-exp-03-25

./scripts/run.sh eval gemini-2.5-pro-exp-03-25 Base-Review/gemini-2.5-pro-exp-03-25
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 CR-Agent/gemini-2.5-pro-exp-03-25
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 PR-Agent/gemini-2.5-pro-exp-03-25
./scripts/run.sh eval gemini-2.5-pro-exp-03-25 SWR-Agent/gemini-2.5-pro-exp-03-25

./scripts/run.sh gen base_review deepseek-chat Base-Review/deepseek-chat
./scripts/run.sh gen cr_agent deepseek-chat CR-Agent/deepseek-chat
./scripts/run.sh gen pr_agent deepseek-chat PR-Agent/deepseek-chat
./scripts/run.sh gen swr_agent deepseek/deepseek-chat SWR-Agent/deepseek-chat

./scripts/run.sh eval deepseek-chat CR-Agent/deepseek-chat
./scripts/run.sh eval deepseek-chat PR-Agent/deepseek-chat
./scripts/run.sh eval deepseek-chat logs/DeepSeek-V3



git clone https://github.com/astropy/astropy.git /SWRBench/data/.cache/astropy__astropy
git clone https://github.com/sphinx-doc/sphinx.git /SWRBench/data/.cache/sphinx-doc__sphinx
git clone https://github.com/pydata/xarray.git /SWRBench/data/.cache/pydata__xarray
git clone https://github.com/sympy/sympy.git /SWRBench/data/.cache/sympy__sympy
git clone https://github.com/scikit-learn/scikit-learn.git /SWRBench/data/.cache/scikit-learn__scikit-learn
git clone https://github.com/pytest-dev/pytest.git /SWRBench/data/.cache/pytest-dev__pytest
git clone https://github.com/psf/requests.git /SWRBench/data/.cache/psf__requests
git clone https://github.com/matplotlib/matplotlib.git /SWRBench/data/.cache/matplotlib__matplotlib
git clone https://github.com/django/django.git /SWRBench/data/.cache/django__django
git clone https://github.com/pylint-dev/pylint.git /SWRBench/data/.cache/pylint-dev__pylint
git clone https://github.com/pallets/flask.git /SWRBench/data/.cache/pallets__flask
git clone https://github.com/mwaskom/seaborn.git /SWRBench/data/.cache/mwaskom__seaborn




./scripts/run_szz.sh /SWRBench/data/astropy__astropy/fix_commits.json /SWRBench/pyszz_v2/conf/bszz_issues_filter.yml /SWRBench/data/.cache/ astropy__astropy
./scripts/run_szz.sh /SWRBench/data/sphinx-doc__sphinx/fix_commits.json /SWRBench/pyszz_v2/conf/bszz_issues_filter.yml /SWRBench/data/.cache/ sphinx-doc__sphinx
./scripts/run_szz.sh /SWRBench/data/pydata__xarray/fix_commits.json /SWRBench/pyszz_v2/conf/bszz_issues_filter.yml /SWRBench/data/.cache/ pydata__xarray
./scripts/run_szz.sh /SWRBench/data/sympy__sympy/fix_commits.json /SWRBench/pyszz_v2/conf/bszz_issues_filter.yml /SWRBench/data/.cache/ sympy__sympy
./scripts/run_szz.sh /SWRBench/data/scikit-learn__scikit-learn/fix_commits.json /SWRBench/pyszz_v2/conf/bszz_issues_filter.yml /SWRBench/data/.cache/ scikit-learn__scikit-learn
./scripts/run_szz.sh /SWRBench/data/pytest-dev__pytest/fix_commits.json /SWRBench/pyszz_v2/conf/bszz_issues_filter.yml /SWRBench/data/.cache/ pytest-dev__pytest
./scripts/run_szz.sh /SWRBench/data/psf__requests/fix_commits.json /SWRBench/pyszz_v2/conf/bszz_issues_filter.yml /SWRBench/data/.cache/ psf__requests
./scripts/run_szz.sh /SWRBench/data/matplotlib__matplotlib/fix_commits.json /SWRBench/pyszz_v2/conf/bszz_issues_filter.yml /SWRBench/data/.cache/ matplotlib__matplotlib
./scripts/run_szz.sh /SWRBench/data/django__django/fix_commits.json /SWRBench/pyszz_v2/conf/bszz_issues_filter.yml /SWRBench/data/.cache/ django__django
./scripts/run_szz.sh /SWRBench/data/pylint-dev__pylint/fix_commits.json /SWRBench/pyszz_v2/conf/bszz_issues_filter.yml /SWRBench/data/.cache/ pylint-dev__pylint
./scripts/run_szz.sh /SWRBench/data/pallets__flask/fix_commits.json /SWRBench/pyszz_v2/conf/bszz_issues_filter.yml /SWRBench/data/.cache/ pallets__flask
./scripts/run_szz.sh /SWRBench/data/mwaskom__seaborn/fix_commits.json /SWRBench/pyszz_v2/conf/bszz_issues_filter.yml /SWRBench/data/.cache/ mwaskom__seaborn



git clone https://github.com/pandas-dev/pandas.git /SWRBench/data_train/.cache/pandas-dev__pandas
git clone https://github.com/numpy/numpy.git /SWRBench/data_train/.cache/numpy__numpy
git clone https://github.com/pallets/click.git /SWRBench/data_train/.cache/pallets__click
git clone https://github.com/psf/black.git /SWRBench/data_train/.cache/psf__black
git clone https://github.com/urllib3/urllib3.git /SWRBench/data_train/.cache/urllib3__urllib3
git clone https://github.com/python/mypy.git /SWRBench/data_train/.cache/python__mypy
git clone https://github.com/scipy/scipy.git /SWRBench/data_train/.cache/scipy__scipy
git clone https://github.com/pre-commit/pre-commit.git /SWRBench/data_train/.cache/pre-commit__pre-commit
git clone https://github.com/python-pillow/Pillow.git /SWRBench/data_train/.cache/python-pillow__Pillow
git clone https://github.com/pycqa/flake8.git /SWRBench/data_train/.cache/pycqa__flake8
git clone https://github.com/pypa/setuptools.git /SWRBench/data_train/.cache/pypa__setuptools
git clone https://github.com/pydantic/pydantic.git /SWRBench/data_train/.cache/pydantic__pydantic
git clone https://github.com/Textualize/rich.git /SWRBench/data_train/.cache/Textualize__rich
git clone https://github.com/aio-libs/aiohttp.git /SWRBench/data_train/.cache/aio-libs__aiohttp
git clone https://github.com/boto/boto3.git /SWRBench/data_train/.cache/boto__boto3
git clone https://github.com/networkx/networkx.git /SWRBench/data_train/.cache/networkx__networkx
git clone https://github.com/giampaolo/psutil.git /SWRBench/data_train/.cache/giampaolo__psutil
git clone https://github.com/huggingface/transformers.git /SWRBench/data_train/.cache/huggingface__transformers
git clone https://github.com/pallets/jinja.git /SWRBench/data_train/.cache/pallets__jinja
git clone https://github.com/pyca/cryptography.git /SWRBench/data_train/.cache/pyca__cryptography
git clone https://github.com/ipython/ipython.git /SWRBench/data_train/.cache/ipython__ipython
git clone https://github.com/tqdm/tqdm.git /SWRBench/data_train/.cache/tqdm__tqdm
git clone https://github.com/opencv/opencv-python.git /SWRBench/data_train/.cache/opencv__opencv-python
git clone https://github.com/python-attrs/attrs.git /SWRBench/data_train/.cache/python-attrs__attrs
git clone https://github.com/theskumar/python-dotenv.git /SWRBench/data_train/.cache/theskumar__python-dotenv
git clone https://github.com/cython/cython.git /SWRBench/data_train/.cache/cython__cython
git clone https://github.com/nedbat/coveragepy.git /SWRBench/data_train/.cache/nedbat__coveragepy
git clone https://github.com/benjaminp/six.git /SWRBench/data_train/.cache/benjaminp__six
git clone https://github.com/nodejs/node-gyp.git /SWRBench/data_train/.cache/nodejs__node-gyp
git clone https://github.com/fastapi/fastapi.git /SWRBench/data_train/.cache/fastapi__fastapi
git clone https://github.com/redis/redis-py.git /SWRBench/data_train/.cache/redis__redis-py
git clone https://github.com/pytest-dev/pytest-cov.git /SWRBench/data_train/.cache/pytest-dev__pytest-cov
git clone https://github.com/scrapy/scrapy.git /SWRBench/data_train/.cache/scrapy__scrapy
git clone https://github.com/gitpython-developers/GitPython.git /SWRBench/data_train/.cache/gitpython-developers__GitPython
git clone https://github.com/streamlit/streamlit.git /SWRBench/data_train/.cache/streamlit__streamlit
git clone https://github.com/python/typing_extensions.git /SWRBench/data_train/.cache/python__typing_extensions
git clone https://github.com/python-jsonschema/jsonschema.git /SWRBench/data_train/.cache/python-jsonschema__jsonschema
git clone https://github.com/jpadilla/pyjwt.git /SWRBench/data_train/.cache/jpadilla__pyjwt
git clone https://github.com/pypa/twine.git /SWRBench/data_train/.cache/pypa__twine
git clone https://github.com/yaml/pyyaml.git /SWRBench/data_train/.cache/yaml__pyyaml
git clone https://github.com/ray-project/ray.git /SWRBench/data_train/.cache/ray-project__ray
git clone https://github.com/keras-team/keras.git /SWRBench/data_train/.cache/keras-team__keras
git clone https://github.com/plotly/plotly.py.git /SWRBench/data_train/.cache/plotly__plotly.py
git clone https://github.com/mongodb/mongo-python-driver.git /SWRBench/data_train/.cache/mongodb__mongo-python-driver
git clone https://github.com/pytorch/vision.git /SWRBench/data_train/.cache/pytorch__vision    
git clone https://github.com/lxml/lxml.git /SWRBench/data_train/.cache/lxml__lxml
git clone https://github.com/scikit-image/scikit-image.git /SWRBench/data_train/.cache/scikit-image__scikit-image
git clone https://github.com/encode/httpx.git /SWRBench/data_train/.cache/encode__httpx
git clone https://github.com/joke2k/faker.git /SWRBench/data_train/.cache/joke2k__faker
git clone https://github.com/pygments/pygments.git /SWRBench/data_train/.cache/pygments__pygments
git clone https://github.com/openai/openai-python.git /SWRBench/data_train/.cache/openai__openai-python
