#!/bin/bash

# preparing local repo-directory
unzip test/repos_test.zip -d test/

# executing MA-SZZ switching back to main pyszz directory
# in this case, the bug-fixes.json contains bugfix commits without issue date
./run_docker.sh /hdd1/zzr/SWRBench/pyszz_v2/test/bugfix_commits_test.json /hdd1/zzr/SWRBench/pyszz_v2/conf/maszz.yml /hdd1/zzr/SWRBench/pyszz_v2/test/repos_test/

# cleanup
rm -rf test/repos_test