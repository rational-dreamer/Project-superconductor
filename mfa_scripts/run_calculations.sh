#!/bin/bash
python task_parser.py "$1" | while read -r line; do
params=($line)
wolfram_params="${params[@]:0:7}"
echo "Running calculation with parameters: $wolfram_params"

wolframscript phase_diagrammer.wls $wolfram_params

if [ $? -ne 0 ]; then
        echo "Error: Calculation failed for params: $wolfram_params"
        fi		
done

echo "All calculations completed."