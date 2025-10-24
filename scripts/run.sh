#!/bin/bash

# set -x PYTHONPATH $PYTHONPATH (pwd)

export OPENAI_API_BASE=your_openai_api_base_url
export OPENAI_API_KEY=your_openai_api_key


workdir=$(dirname $(dirname $0))
echo $workdir

run_base_review() {
    dataset=$1
    model=$2
    logs_dir=$3
    num_threads=$4
    mkdir -p $logs_dir

    python swrbench/generation.py \
        --dataset-file $dataset \
        --model $model \
        --max-tokens 8192 \
        --temperature 0.2 \
        --num-threads $num_threads \
        --output-file $logs_dir/generation.jsonl

}

run_hybrid_review() {
    dataset=$1
    model=$2
    logs_dir=$3
    num_threads=$4
    mkdir -p $logs_dir

    python swrbench/hybrid_review.py \
        --dataset-file $dataset \
        --model $model \
        --temperature 0.2 \
        --num-threads $num_threads \
        --output-file $logs_dir/generation.jsonl

}

run_refine_review() {
    dataset=$1
    model=$2
    logs_dir=$3
    num_threads=$4
    mkdir -p $logs_dir

    python swrbench/generation.py \
        --dataset-file $dataset \
        --model $model \
        --refine \
        --max-tokens 8192 \
        --temperature 0.2 \
        --num-threads $num_threads \
        --output-file $logs_dir/generation.jsonl

}


run_npr_refine_review() {
    dataset=$1
    model=$2
    logs_dir=$3
    num_threads=$4
    review_paths=$5
    mkdir -p $logs_dir

    python swrbench/npr_review.py \
        --dataset-file $dataset \
        --model $model \
        --refine \
        --temperature 0.2 \
        --num-threads $num_threads \
        --output-file $logs_dir/generation.jsonl \
        --review-paths $review_paths

}

run_npr_review() {
    dataset=$1
    model=$2
    logs_dir=$3
    num_threads=$4
    mkdir -p $logs_dir

    python swrbench/npr_review.py \
        --dataset-file $dataset \
        --model $model \
        --max-tokens 32000 \
        --temperature 0.2 \
        --num-threads $num_threads \
        --output-file $logs_dir/generation.jsonl

}

run_cr_agent() {
    dataset=$1
    model=$2
    logs_dir=$3
    num_threads=$4
    mkdir -p $logs_dir

    python scripts/run_cr_agent.py \
        --dataset-file $dataset \
        --api-key $OPENAI_API_KEY \
        --base-url $OPENAI_API_BASE \
        --model $model \
        --num-threads $num_threads \
        --output-dir $logs_dir 

}

run_pr_agent() {
    dataset=$1
    model=$2
    logs_dir=$3
    num_threads=$4
    mkdir -p $logs_dir

    python swrbench/pr_agent.py \
        --dataset-file $dataset \
        --model $model \
        --temperature 0.2 \
        --num-threads $num_threads \
        --output-file $logs_dir/generation.jsonl

}

# run_pr_agent() {
#     dataset=$1
#     model=$2
#     logs_dir=$3
#     num_threads=$4
#     mkdir -p $logs_dir

#     python scripts/run_pr_agent.py \
#         --dataset-file $dataset \
#         --api-key $OPENAI_API_KEY \
#         --base-url $OPENAI_API_BASE \
#         --temperature 0.2 \
#         --model $model \
#         --num-threads $num_threads \
#         --output-dir $logs_dir 

# }

run_swr_agent() {
    export LITELLM_LOCAL_MODEL_COST_MAP=True

    dataset=$1
    model=$2
    logs_dir=$3
    mkdir -p $logs_dir

    docker container ls -a | grep swe-agent-docker | awk '{print $1}' | xargs docker rm -f

    python scripts/run_swr_agent.py \
        --dataset-file $dataset \
        --api-key $OPENAI_API_KEY \
        --base-url $OPENAI_API_BASE \
        --model $model \
        --num-threads 16 \
        --output-dir $logs_dir \

        
}

eval() {
    export OPENAI_API_KEY=your_openai_api_key
    export OPENAI_API_BASE=your_openai_api_base_url

    dataset=$1
    judge_model=$2
    result_dir=$3
    python swrbench/evaluation_struct.py \
        --model $judge_model \
        --num-threads 32 \
        --dataset-file $dataset \
        --pred-file $result_dir/generation.jsonl \
        --output-file $result_dir/evaluation__${judge_model}.json 

}

task=$1
# dataset=/SWRBench/data/swr_datasets_0520_d3c3.jsonl
# dataset_name=swr_datasets_0520_d3c3
dataset=/SWRBench/data/swr_datasets_0520_d5c5.jsonl
dataset_name=swr_datasets_0520_d5c5
if [ $task == "gen" ]; then
    baseline=$2
    model_name=$3
    run_id=$4

    logs_dir=$workdir/logs/$dataset_name/$run_id
    mkdir -p $logs_dir

    printf "run_id: %s\n" $run_id
    printf "baseline: %s\n" $baseline
    printf "model_name: %s\n" $model_name
    printf "logs_dir: %s\n" $logs_dir
    
    num_threads=32
    if [[ $model_name == "gemini-2.0-flash-exp" || $model_name == "openai/gemini-2.0-flash-exp" ]]; then
        num_threads=1
    elif [[ $model_name == "gemini-2.0-flash-thinking-exp" || $model_name == "openai/gemini-2.0-flash-thinking-exp" ]]; then
        num_threads=1
    elif [[ $model_name == "gpt-4o" ]]; then
        export OPENAI_API_BASE=your_openai_api_base_url
        export OPENAI_API_KEY=your_openai_api_key
        num_threads=4
    fi

    printf "num_threads: %s\n" $num_threads

    if [ $baseline == "base_review" ]; then
        run_base_review $dataset $model_name $logs_dir $num_threads
    elif [ $baseline == "refine_review" ]; then
        run_refine_review $dataset $model_name $logs_dir $num_threads
    elif [ $baseline == "cr_agent" ]; then
        run_cr_agent $dataset $model_name $logs_dir $num_threads
    elif [ $baseline == "pr_agent" ]; then
        run_pr_agent $dataset $model_name $logs_dir $num_threads
    elif [ $baseline == "npr_review" ]; then
        run_npr_review $dataset $model_name $logs_dir $num_threads
    elif [ $baseline == "npr_refine_review" ]; then
        review_paths=$5
        printf "review_paths: %s\n" $review_paths
        run_npr_refine_review $dataset $model_name $logs_dir $num_threads $review_paths
    elif [ $baseline == "swr_agent" ]; then
        run_swr_agent $dataset $model_name $logs_dir
    elif [ $baseline == "hybrid_review" ]; then
        run_hybrid_review $dataset $model_name $logs_dir $num_threads
    else
        echo "Invalid baseline"
    fi
elif [ $task == "eval" ]; then
    judge_model=$2
    run_id=$3
    logs_dir=$workdir/logs/$dataset_name/$run_id
    eval $dataset $judge_model $logs_dir
fi
