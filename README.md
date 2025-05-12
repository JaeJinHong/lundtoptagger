# lundtoptagger

## Setup

On UChicago, samples and flat weights are here:  
`/data/jmsardain/LJPTagger/FullSplittings/SplitForTopTagger/`

The included setup script can set up the environment on several different systems:

- a system with Red Hat Enterprise Linux 9, an NVIDIA driver which supports CUDA >= 11.8, and access to CVMFS, such as `lxplus-gpu`
- a system with CentOS 7 and access to CVMFS (currently set up without CUDA)
- UCL's `gpu02` server

The script will automatically figure out which of these systems it is running on and set up the environment accordingly; just do

```bash
source setup.sh
```

On UChicago, do

```bash
source /data/jmsardain/LJPTagger/JetTagging/miniconda/bin/activate
conda activate rootenv
```

## Data preparation

To create graphs for training from ROOT files and save them to a file, run

```bash
python Make_data.py configs/config_make_data.yaml
```

## Training and testing

For the training, the main changes one should do are in the configuration file: `config_ONLY_TRAIN.yaml`.
In this file you will define the learning rate, batch size, the input files, the model to use, the location to save your checkpoints.

To run the training:

```bash
python weight_ONLY_TRAINS.py configs/config_ONLY_TRAIN.yaml
```

You can override the $k_T$ cut in the configuration file using a command line argument:

```bash
python weight_ONLY_TRAINS.py configs/config_ONLY_TRAIN.yaml --ln_kT_cut 0
```


For the testing, you should run the final_makescores notebook. The only changes you should do are under the conditions in the for loop. The different variables should point to your test files, the ckpt you want to use and the repo to save your output root files 

At the end, when you are done with the testing, make sure you hadd all the root files together: 
```
hadd -f tree.root user.*root
```

## Plotting

In a clean and new terminal, go to the plotting repo and source the setup file. 
It will get the version of the libraries you want to use from /cvmfs/. 
Go to plotting.py and check that you are using the root file you just created with hadd after the testing of the model. 
Plot! 
```
source setup.sh
python -b plotting.py 
```

## To do list: 
- [ ] Cut on ln(kt): prepare multiple graphs with different values of ln(kT) cuts 
- [ ] Make a bkg rej vs ln(kT) plot
- [ ] Make the LundJetPlane plot with the prediction to see where the modeling uncertainties impact the most
- [ ] Apply a shift of 5% to mean pT of the constituent, and test on that sample
- [ ] Apply a shift of 5% to resolution pT of the constituent, and test on that sample
