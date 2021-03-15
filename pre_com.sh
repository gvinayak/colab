#!/bin/sh 
rm Pre_Files/* 
python2 precompute_text.py 0 23 &
python2 precompute_text.py 23 46 &
python2 precompute_text.py 46 69 &
python2 precompute_text.py 69 end 
wait
cat $(find ./Pre_Files/ -name "pre_compute_map_*" | sort -V) > ./Pre_Files/pre_compute_map.txt 
cat $(find ./Pre_Files/ -name "pre_compute_Aij_*" | sort -V) > ./Pre_Files/pre_compute_Aij.txt
