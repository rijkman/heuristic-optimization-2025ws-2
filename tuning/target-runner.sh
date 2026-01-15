#!/bin/bash
error() {
    echo "`TZ=UTC date`: $0: error: $@"
    exit 1
}

# read command line arguments
SCRIPT=$1
CONFIG_ID=$2
INSTANCE_ID=$3
SEED=$4
INSTANCE=$5
shift 5 || error "Not enough parameters"
CONFIG_PARAMS=$*

# define executable paths
export LD_LIBRARY_PATH="" # clean deps
EXE=/root/.juliaup/bin/julia # /usr/local/bin/julia
EXE_PARAMS="$SCRIPT --instance $INSTANCE --seed ${SEED} ${CONFIG_PARAMS}"

if [ ! -x "$(command -v ${EXE})" ]; then
    error "${EXE}: not found or not executable (pwd: $(pwd))"
fi

STDOUT=c${CONFIG_ID}-${INSTANCE_ID}-${SEED}.stdout
STDERR=c${CONFIG_ID}-${INSTANCE_ID}-${SEED}.stderr

$EXE ${EXE_PARAMS} 1> ${STDOUT} 2> ${STDERR}

if [ ! -s "${STDOUT}" ]; then
    error "${STDOUT}: No such file or directory"
fi

# read output with format: {cost} {time}
COST=$(tail -n 1 ${STDOUT} | grep -e '^[[:space:]]*[+-]\?[0-9]' | cut -f1)
echo "$COST"
rm -f "${STDOUT}" "${STDERR}"
exit 0