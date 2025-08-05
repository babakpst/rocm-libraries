#!/bin/bash

NAME="BBS_NT_0"
YAML="$NAME.yaml"
OUT="$NAME-tensilelite.log"

WORK_DIR="WDirDevice_id"

echo "running $NAME ..."
/home/bpoursar/workspace/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/bin/Tensile $YAML $WORK_DIR 2>&1 | tee $OUT
rm -rf $WORK_DIR/1\_BenchmarkProblems 
mkdir build_BBS_NT_0
cp  $YAML build_BBS_NT_0
mv $WORK_DIR/2\_BenchmarkData $WORK_DIR/3\_LibraryLogic $OUT build_BBS_NT_0
echo " ---- $NAME Done!"
