#!/bin/bash
#SBATCH -J DataExtractJob
#SBATCH -N 1
#SBATCH -o /netscratch/kumar/data_handler/DataExtractJob-%j.out
#SBATCH -e /netscratch/kumar/test/DataExtractJob-%j.err
#SBATCH -t 30
#SBATCH --mail-type=END

echo "Executing on $HOSTNAME"
python extract_coco_metainfo.py
