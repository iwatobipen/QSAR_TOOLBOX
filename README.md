# QASR_TOOLBOX

Bunch of scripts for QSAR model building.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Prerequisites

Numpy, Scikit-learn, Pandas optuna and RDKit
I recommend to install packages avobe with conda.

```
conda install -c conda-forge pandas
conda install -c conda-forge rdkit
```


## BASIC USAGE

### FPGENERATION SCRIPT


```
usage: fpgen.py [-h] [--fptype FPTYPE] [--radius RADIUS] [--nBits NBITS]
                [--molid MOLID] [--target TARGET] [--output OUTPUT]
                input

positional arguments:
  input            finename of sdf

optional arguments:
  -h, --help       show this help message and exit
  --fptype FPTYPE  ECFP, FCFP,
  --radius RADIUS  radius of ECFP, FCFP
  --nBits NBITS    number of bits
  --molid MOLID    molid prop
  --target TARGET  target name for predict
  --output OUTPUT  output path

```

```
python qsartools/fpgen.py example/CHEMBL25-chembl_activity-JAK3.sdf --molid Molecule --target Class

```


### BUILD MODEL AND OPTIMIZATION



```
usage: svm_rf_gp_optuna.py [-h] [--testsize TESTSIZE] [--n_trials N_TRIALS]
                           [--outpath OUTPATH]
                           X Y target

positional arguments:
  X                    data for X npz format
  Y                    data for Y npz format
  target               name of target, it will be database name

optional arguments:
  -h, --help           show this help message and exit
  --testsize TESTSIZE  testsize of train/test split default = 0.2
  --n_trials N_TRIALS  number of trials default = 200
  --outpath OUTPATH    path for output default is lgb_output

```

```
python qsartools/svm_rf_gp_optuna.py data/ECFP_2_1024_X.npz data/Class_arr.npz JAK3_ --n_trial 50
```


## Authors

* **@iwatobipen**

## License

This project is licensed under the BSD License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

