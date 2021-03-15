#!/bin/sh 
rm Pre_Files/* 
python2 precompute_text.py 0 5 &
python2 precompute_text.py 5 10 &
python2 precompute_text.py 10 15 &
python2 precompute_text.py 15 end 
wait
cat $(find ./Pre_Files/ -name "pre_compute_map_*" | sort -V) > ./Pre_Files/pre_compute_map.txt 
cat $(find ./Pre_Files/ -name "pre_compute_Aij_*" | sort -V) > ./Pre_Files/pre_compute_Aij.txt
