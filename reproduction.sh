#!/bin/bash

## Run analytical annealing for two different data fractions [0.1, 0.2] and
## number of angle segments [32, 64]
for i in {0..31}; do kedro run --pipeline=adiabatic_trackrec --params="num_angle_parts=32,data_fraction=0.1,geometric_index=$i"; done
for i in {0..63}; do kedro run --pipeline=adiabatic_trackrec --params="num_angle_parts=64,data_fraction=0.1,geometric_index=$i"; done
for i in {0..63}; do kedro run --pipeline=adiabatic_trackrec --params="num_angle_parts=64,data_fraction=0.2,geometric_index=$i"; done

## Remove empty results files (these exist, as we do not try to compute
## instances with > 23 and 0 variables).
find data/04_adiabatic/ -size 1 -type f -delete
find data/04_adiabatic/ -type d -empty -delete

## Run QAOA optimisation and compute anneal schedules.
kedro run --pipeline=qaoa_trackrec --params="max_p=50,geometric_index=27,data_fraction=0.1,num_angle_parts=64,q=-1,initialisation=first_random" # RI
for q in {1,5,10,25,50}; do kedro run --pipeline=qaoa_trackrec --params="max_p=50,geometric_index=27,data_fraction=0.1,num_angle_parts=64,q=$q,initialisation=zeros"; done
