#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
OUTDIR=$SCRIPTPATH/output

./build.sh

# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
MEM_LIMIT="12g"

# create output dir if it does not exist
if [ ! -d $OUTDIR ]; then
  mkdir $OUTDIR;
fi

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/:/input/ \
        -v $OUTDIR:/output/ \
        hanseg2023algorithm
