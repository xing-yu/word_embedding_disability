#!/bin/bash

# Request processors (openmp * mpi)
#PBS -l nodes=1:ppn=32
#PBS -l walltime=100:00:00
#PBS -l gres=ccm

# Export all my environment variables to the job
#PBS -V

# Queue name (see info about other queues in web documentation)
#PBS -q cpu 

# Send mail when the job begins and ends (optional)
#PBS -m be

#PBS -j oe
#------------------------------
# End of embedded Qcase 


module load ccm
module load anaconda3
source activate word2vec
cd /N/u/yu64/BigRed2/word_embedding_disability
ccmrun python3 ./source/word2vec.py ./data/dic.json embedding.txt pretrains_100d.txt 128 5 100 32 ./checkpoints 10000
source deactivate
