#!/bin/bash

# Create log directory
mkdir -p logs

# Run all experiments sequentially and redirect logs
python case_homo.py > logs/case_homo.log 2>&1
python case_hete.py > logs/case_hete.log 2>&1
python case_hete_d.py > logs/case_hete_d.log 2>&1

python case_global_homo.py > logs/case_global_homo.log 2>&1
python case_global_hete.py > logs/case_global_hete.log 2>&1
python case_global_hete_d.py > logs/case_global_hete_d.log 2>&1

# Wait for all background processes to finish
wait

echo "All experiments have completed."