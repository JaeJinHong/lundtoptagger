# lundtoptagger

samples and flat weights are here : 
```
/data/jmsardain/LJPTagger/FullSplittings/SplitForTopTagger/ 
```

For the training, the main changes one should do are in the configuration file: configs/config_class_train_top.yaml .
In this file you will define the learning rate, batch size, the input files, the model to use, the repo to save your ckpts. 

To run the training: 
```
## if you are not working on UChicago, do not do the first 2 lines
source /data/jmsardain/LJPTagger/JetTagging/miniconda/bin/activate
conda activate rootenv

python weight_class_train.py configs/config_class_train_top.yaml
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

This plotting code is a bit old, it uses root_numpy librarie (which is not supported by newer versions of root), so it is necessary to use centos7. The ideal would be to replace all root_numpy functions for native root functions or other libraries.

```
setupATLAS -c centos7
source setup.sh
python plotting.py 
```

