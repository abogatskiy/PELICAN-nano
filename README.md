# PELICAN Nano Network for Particle Physics

Stripped down version of PELICAN for training ultra-lightweight and interpretable top taggers.
Modified from original PELICAN code: https://arxiv.org/abs/2211.00454
The command used to train PELICAN Nano can be found [below](#executing-the-training-scripts)

## Anonymous Authors

### Dependencies

* python >=3.9
* pytorch >=1.10
* h5py
* colorlog
* scikit-learn
* tensorboard (if using --summarize)
* optuna (if using the optuna script)
* psycopg2-binary (if using optuna with a distributed file sharing system)

### Installing

* No installation required -- just use one of the top level scripts.

### General Usage Tips

* The classifier can be used via the script `train_pelican_classifier.py`.
    The main required argument is `--datadir` since it provides the datasets. 
* See data/sample_data/ for a small example of datasets. Each datapoint contains some number of input 4-momenta (E,p_x,p_y,p_z) under the key `Pmu`, 
    target 4-momenta (e.g. the true top momentum) under `truth_Pmu` and a classification label under `is_signal`. The choice of the target key is controlled by the argument `--target`.
* The same script can be used for inference on the test dataset when it is run with the flag `--eval`. 
* Model checkpoints can be loaded via `--load` for inference or continued training. 
* By default the model is run on a GPU, but CPU evaluation can be forced via `--cpu`.
* The argument `--verbose` (correspodinginly `--no-verbose`) can be used to write per-minibatch stats into the log.
* The number of particles in each event can vary. Most computations are done with masks that properly exclude zero 4-momenta. The argument `--nobj` sets the maximum number of particles loaded from the dataset. Argument `--add-beams` also appends two "beam" particles of the form (1,0,0,±1) to the inputs to help the network learn that bias in the dataset due to the fixed z-axis. With this flag, the intermediate tensors in Eq2to2 layers will have shape [Batch, 2+Nobj, 2+Nobj, Channel].

### Outputs of the script

* Logfiles are kept in the `log/` folder. By default these contain initializtion information followed by training and validation stats for each epoch, and testing results at the end. The argument `--prefix` is used to name all files. Re-running the script without changing the prefix will overwrite all output files unless the flag `--load` is used.
* If `--summarize` is on, then Tensorboard summaries are saved into a folder with the same name as the log. If `--summarize-scv` is 'all', then per-minibatch stats are written into a separate CSV file. If it's 'test', then only the stats from the testing dataset are saved.
* At the end of evaluation on the testing set, the stats are written into a CSV file whose name ends in `Best.metrics.csv` (for the model checkpoint with the best validation score) and `Final.metrics.csv` (for the model checkpoint from the last epoch). If there are multiple runs whose prefixes only differ by text after the last dash (e.g. `run-1, run-2, run-3`, etc.) then their metrics will be appended to the same CSV.
* Model outputs (predictions) are saved as .pt files in `predict/`
* Model checkpoints are saved as .pt files in `model/`.

### Executing the training scripts

* Here is an example of a command that starts training the classifier on the sample dataset that is part of this repository. Optimally there should be three files with names train.h5, valid.h5, and test.h5. Add --cpu or --cuda to choose device.
```
python3 train_pelican_nano.py --datadir=./data/sample_data --target=is_signal --nobj=80 --nobj-avg=49 --num-epoch=140 --num-train=-1 --num-valid=20000 --batch-size=256 --prefix=test --drop-rate=0.05 --drop-rate-out=0.05 --weight-decay=0.005
```

* Here is the command used to train PELICAN Nano on the toptag dataset (can be downloaded from https://osf.io/7u3fk/?view_only=8c42f1b112ab4a43bcf208012f9db2df, but rename the validation file to contain 'valid')
```
python3 train_pelican_nano.py --datadir=./data/toptag --target=is_signal --n-hidden=1 --nobj=80 --num-epoch=140 --num-train=-1 --num-valid=20000 --batch-size=256 --prefix=test --drop-rate=0.05 --drop-rate-out=0.05 --weight-decay=0.005
```


## Original PELICAN Authors

Alexander Bogatskiy, Flatiron Institute

Jan T. Offermann, University of Chicago 

Timothy Hoffman, University of Chicago

Xiaoyang Liu, University of Chicago

## Acknowledgments

Inspiration, code snippets, etc.
* [Masked BatchNorm](https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py)
* [Gradual Warmup Scheduler](https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py)
* [whichcraft](https://github.com/cookiecutter/whichcraft)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details