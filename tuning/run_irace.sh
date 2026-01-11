#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N irace_test
#$ -l h_vmem=4G
source /home1/share/conda/miniforge3/etc/profile.d/conda.sh
conda activate r-irace
Rscript -e 'library(irace); irace.cmdline()'
