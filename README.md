

## MFA
[Link](https://drive.google.com/drive/folders/1TeT_Ut5pPm3CNHSIqgvA0vpjtnqufFRQ?usp=drive_link) to a dataset of prepared resized MFA phase diagram images.

The mfa_scripts folder contains the following files:
1) phase_diagrammer.wls - WM script that accepts the Hamiltonian parameters D, V, J, tb, tp, tn, tpn and produces a phase diagram.
Example run:
```
phase_diagrammer.wls 0.12 0.28 1 0.5 0.4 0.4 0.05
```
2) task_parser.py - processes the sample_params.json task file
3) run_calculations.sh - runs calculations via a JSON file:
```
run_calculations.sh sample_params.json
```

##Heat bath
[Link](https://drive.google.com/file/d/17eCNfKXNzxqwaY9UYharl6LsZmYsN6A3/view?usp=drive_link) to a CSV zip of calculated data on
[Link](https://drive.google.com/drive/folders/1DrVuy_JWwjr5hKHdfcl8E4CCwVDHwBVq?usp=drive_link) to a dataset of prepared resized cluster1 phase diagram images.

