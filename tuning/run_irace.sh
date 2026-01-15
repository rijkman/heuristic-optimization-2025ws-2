#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N irace_test
#$ -l h_vmem=4G

# read command line arguments
ALGORITHM=$1
INSTANCE_SIZE=$2

# activate work environment
source /home1/share/conda/miniforge3/etc/profile.d/conda.sh
conda activate r-irace

# execute irace tuning
Rscript irace_runner.r $ALGORITHM $INSTANCE_SIZE
Rscript irace_parser.r $ALGORITHM $INSTANCE_SIZE
