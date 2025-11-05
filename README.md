## Machine learning models

The UNET folder contains the following files:
1) processing_data - image preparation (compression from ~500x500 pixels to 224x224), creating a CSV file with markup, and splitting this file into three (train, val, test). There are two versions: for MFA and cluster1.
2) UNET_FULL_2 - auxiliary UNET model (with encoder and decoder), heavily overtrained, functions as an autoencoder. At the end of training, it generates saved encoder weights in the encoder_weights.npy file;
3) UNET_WE_5 - the model itself. Optionally, you can freeze the encoder at the end so that it saves the FULL model weights;
4) dataset creation - Recreates the dataset from the folder containing the source CSV files and the "results" directory with the generated images to the "output_dataset_directory" folder, which will be used for training and testing;
5) dataset_common_functions - an auxiliary script for running dataset_creation;
6) test_model - the actual model metrics;
7) parameter_distribution - to construct histograms of parameter distribution.

Launch order: 1, 2, 3, 4, 6.

For convenience, everything is stored in config.json:
- "resize_images": true/false - whether to compress images;
- "input_dir" - path to the folder with the source uncompressed images (ignored if resize_images = false);
- "output_dir" - path to the folder with the prepared images;
- "learning_dir" - path to the folder with the training data.

**Launch of UNET training:**
```
./learn_UNET_mfa.sh config.json
```
For MFA data or
```
./learn_UNET_cluster1.sh config.json
```
For heat bath data (they differ only in the order of the parameters in the file names)

### VGG16 BN  
The VGG16BN folder contains the following files: 
1. processing_data.py - image preparation (compression from ~500x500 pixels to 224x224), creation of a csv file with markup, splitting this file into 3 (train, val, test). There are two versions: for MFA and cluster1;
2. dataset.py - dataset pre-preparation;
3. VGG16BN.py - the model itself;
4. train_model.py - model training; after training is complete, we obtain a graph of training and validation losses;
5. test_model.py - model testing; after testing, saves a csv file with predictions;
6. stats.py - calculation of the relative detection error; statistical processing of results (min, max, mean) with console output Creates a CSV file with the following columns:
* True parameter values ​​(D, V, tb, tp)
* Predicted parameter values ​​(Dpr, Vpr, tbpr, tppr)
* Calculated MAPE values ​​(MAPE_D, MAPE_V, MAPE_tb, MAPE_tp)

Launch order: 1, 4, 5, 6

For convenience, everything is stored in config.json:
- "resize_images": true/false - whether to compress images;
- "input_dir" - path to the folder with the source uncompressed images (ignored if resize_images = false);
- "output_dir" - path to the folder with the prepared images;
- "learning_dir" - path to the folder with the training data.

**Launch of training:**
```
./learn_VGG_mfa.sh config.json
```
For MFA data or
```
./learn_VGG_cluster1.sh config.json
```
For heat bath data (they differ only in the order of the parameters in the file names)

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

