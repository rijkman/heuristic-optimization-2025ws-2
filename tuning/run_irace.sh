#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N irace_test
#$ -l h_vmem=4G

# read command line arguments
ALGORITHM=$1
INSTANCE_SIZE=$2

# execute irace tuning
Rscript ./tuning/irace_runner.r $ALGORITHM $INSTANCE_SIZE
Rscript ./tuning/irace_parser.r $ALGORITHM $INSTANCE_SIZE
