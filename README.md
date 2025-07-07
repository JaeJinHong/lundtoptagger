# lundtoptagger

Tag top and W jets using the LundNet model.


## Setup

On UChicago, samples and flat weights are here:  
`/data/jmsardain/LJPTagger/FullSplittings/SplitForTopTagger/`

The included setup script can set up the environment on several different systems:

- a system with Red Hat Enterprise Linux 9, an NVIDIA driver which supports CUDA >= 11.8, and access to CVMFS, such as `lxplus-gpu`
- a system with CentOS 7 and access to CVMFS (currently set up without CUDA)
- UCL's `gpu02` server and Hypatia GPU partition

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

To create graphs for training from ROOT files and save them to a file, first process JETM2 or FTAG1 derivations with the following code:  
<https://gitlab.cern.ch/rvinasco/jetmdatamc/-/tree/temporaryRun2>  
Then run `Make_data.py` on the output:

```bash
python Make_data.py configs/config_make_data.yaml
```

The script applies selections defined in the configuration files, creates Lund trees (graphs) for each jet, and calculates weights which make the jet $p_T$ distribution flat.
Two files are created:
a file containing a list of graphs (`torch_geometric.data.Data` objects) that can be used for training and testing the tagging model,
and a ROOT file containing some properties of the jets passing selection:

- DSID (MC channel number) of the dataset from the which the jet was taken
- MC event weight
- mass, $p_T$, $\eta$, $\phi$, and number of charged constituents of the jets
- weight which makes the $p_T$ distribution flat
- large-R jet truth labels (1 for top, 2 for W, 10 for QCD)
- signal/background label (1 for signal, 0 for background)
- GN2X scores (if available)

Some of these are already stored as attributes of the graphs, and they could all be, but the graphs can take a long time to load,
so it can be useful to have a separate file for plots which don't require the Lund trees.

### Configuration and parameters for `Make_data.py`

The configuration is defined in `configs/config_make_data.yaml`. The path to this config file must be given as a command-line argument, as in the example above.

In this file, you can set:

- the input and output file paths,
- fractions of the data to save in separate files - this can be used for train/test splits, or memory management,
as the events are loaded and processed in chunks of sizes determined by these fractions
- a value for the optional $k_T$ cut

The script uses another configuration file, `config_signal.yaml`, which contains parameter sets for several signal samples.
The path to this file and the choice of parameter set from it are also specified in `config_make_data.yaml` under the `signal_config_file` and `signal` keys, respectively.
The parameters in `config_signal.yaml` include values for the selection cuts (mass, $p_T$, minimum number of splittings)
and paths to files with histograms of the $p_T$ distributions of the jets, which are used to calculate the $p_T$ weights
so that they are proportional to 1/(bin count).
These histograms are included in the repository; they are located in the `histos` folder.
They can be created with the `make_histos.py` script, which also applies mass and $pT$ cuts from `config_signal.yaml`.

Rather than choosing a single signal configuraion, the `signal` key in `config_make_data.yaml` can also be set to `all`,
in which case the script will combine the selections (to include jets which pass any of the selections)
and store multiple sets of flat-pT weights (in both the graphs and ROOT files), one for each signal configuration.
The main reason for this is that you don't have to process the background jets (QCD) multiple times,
and there is no need to have multiple QCD graphs files with slightly different selections.
Instead, when later using the file, you can choose which set of weights to use and apply corresponding selection cuts.
If saving multple sets of weights, they will be saved with the signal configuration identifier as a suffix in the branch/attribure names.
You can also do this with a single signal configuration by setting `signal_name_in_weight` to `True` in `config_make_data.yaml`,
in order to have matching names between the signal files where you would probably only use 1 configuration and the background files where you might want to use multiple configurations.

You can override any of the parameters in `config_make_data.yaml` using the `--override` command-line argument; for example:

```bash
python Make_data.py configs/config_make_data.yaml --override path_to_rootfiles="/path/to/root/files/*.root" id="QCD" event_fractions="[0.0025, 0.0025]" kT_cut=0.5
```

The values for the override arguments should be in the YAML format - e.g. `null` will be interpreted as `None` and `.inf` as `float('inf')`.


## Training

For the training, the main changes one should do are in the configuration file: `config_ONLY_TRAIN.yaml`.
In this file you will define the learning rate, batch size, the input files, the model to use, the location to save your checkpoints.

To run the training:

```bash
python weight_ONLY_TRAINS.py configs/config_ONLY_TRAIN.yaml
```

There are two optional arguments which can be used to override the values in the config file:

- `--ln_kT_cut`: float
- `--do_combined_training`: value can be true/false, yes/no, 0/1, case insensitive

For example:

```bash
python weight_ONLY_TRAINS.py configs/config_ONLY_TRAIN.yaml --ln_kT_cut 0 --do_combined_training true
```

## Testing

Run the testing:

```bash
python test_make_scores.py configs/config_make_scores.yaml
```

Some paths and names in the configuration file can have placeholders that are replaced by values of other parameters,
namely by the values of `kT_cut` and `sample`.
This makes it easy to run on different samples:
if you keep the paths to your samples and output files the same apart from a part that changes with the sample,
you can just change the `sample` parameter in the config file without having to change 3 different variables
(path_to_test_file, path_to_outdir, and output_name).

These two parameters can be overridden via command-line arguments.
For example:

```bash
python test_make_scores.py configs/config_make_scores.yaml --sample Sherpa_Cluster --ln_kT_cut 0
```

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
