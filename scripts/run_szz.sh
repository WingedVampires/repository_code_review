#!/bin/bash

bugfix_commits_file=$1
conf_file=$2
repos_dir=$3
save_id=$4

out_dir=$(dirname "${bugfix_commits_file}")

echo +++ PARAMS +++
echo bugfix_commits_file=$bugfix_commits_file
echo conf_file=$conf_file
echo repos_dir=$repos_dir
echo save_id=$save_id
echo out_dir=$out_dir

# docker build -t pyszz .
workdir=/SWRBench/pyszz_v2
mkdir -p $workdir/out

# replace with `docker run -d` to run the container in detached mode
docker run -it --rm --entrypoint /bin/bash \
        -v $workdir:/usr/src/app \
        -v $bugfix_commits_file:/usr/src/app/bugfix_commits.json \
        -v $conf_file:/usr/src/app/conf.yml \
        -v $repos_dir:/usr/src/app/cloned \
        pyszz -c "python main.py bugfix_commits.json conf.yml $save_id cloned/"

rm $out_dir/szz_results.json
cp $workdir/out/bic_$save_id.json $out_dir/szz_results.json