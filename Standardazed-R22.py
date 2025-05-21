import argparse
import os.path as osp
import os
import glob
import torch
import awkward as ak
import time
import uproot
import uproot3
import numpy as np

import scipy.sparse as ss
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt



#path_to_file = "/data/jmsardain/LJPTagger/FullSplittings/SplitForTopTagger/5_Signal_and_BKG/*train.root"
#path_to_file = "/data/jmsardain/LJPTagger/FullSplittings/SplitForTopTagger/5_Signal_and_BKG_cutTruth350/*test.root"

#path_to_file = "/data/ravinascos/LundNet/train_test_samples/15train_10test/*train.root"
#intreename = "FlatSubstructureJetTree"

## tests New release
#path_to_file = "/data/ravinascos/Lund_Plane_2024_graphs/TRAIN_1/*train.root"
#path_to_file = "/data/ravinascos/Lund_Plane_2024_graphs/New_Samples/Pythia_TRAIN_05QCD_25Z_25W/*train.root"
path_to_file = "/data/ravinascos/Lund_Plane_2024_graphs/UFO_container/QCD_05/*train.root"

intreename = "AnalysisTree"

files = glob.glob(path_to_file)


all_lund_zs = np.array([])
all_lund_kts = np.array([])
all_lund_drs = np.array([])
all_lund_Ntrk = np.array([])

all_pt = np.array([])
all_weights = np.array([])
all_mass = np.array([])

iii = 0
files_count = 0

for file in files:
    print("Loading file",file)
    files_count += 1
    with uproot.open(file) as infile:
        tree = infile[intreename]
        
        #print("111")
        try:
            DSID = tree["dsid"].array(library="np")
            DSID = DSID[0]
            #print(DSID)
        except:
            continue

        if DSID < 364703 or DSID > 364712 : # R22 bkg < 364713  Z' == 801661:  
            continue
            
        #print("222")
        
        jet_pts_truth = ak.flatten(tree["Truth_LRJ_pt"].array(library="ak"))
        jet_pts = ak.flatten(tree["LRJ_pt"].array(library="ak"))
        #ptweights = tree["fjet_testing_weight_pt"].array(library="np")
        ptweights = np.ones(len(jet_pts))
        jet_ms =  ak.flatten(tree["LRJ_mass"].array(library="ak") )
        #ak.to_numpy(tree["LRJ_mass"].array() ) # [:,0]

        
        #jet_pts_truth = jet_pts_truth[:,0]
        #ptweights = ptweights[:,0]
        #jet_pts = jet_pts[:,0]
        #jet_ms =  jet_ms[:,0]
        #print(jet_pts)
        #print(len(jet_pts_truth))
        #print(len(jet_pts))
        #print(len(ptweights))
        #print(len(tree["Truth_LRJ_pt"].array(library="ak")))
        #print(len(tree["LRJ_pt"].array(library="ak")))
        #print("1111111111111!!") 
        #print(jet_pts[:,0])
        #print(ak.flatten(jet_pts, axis=1))
        #print("2222222222!!")  

        #cut_pt = (jet_pts_truth > 0) #& (ptweights > 150 )
        cut_pt = (jet_pts > 200) #& (ptweights > 150 )
        cut_mass = jet_ms > 0
        cut_pt = (cut_pt) & (cut_mass)
        iii += len( cut_pt[cut_pt] )
        
        jet_pts = jet_pts[cut_pt]
        jet_ms = jet_ms[cut_pt]
        ptweights = ptweights[cut_pt]

        all_pt = np.append(all_pt, ak.to_numpy(jet_pts) )
        all_weights = np.append(all_weights, ak.to_numpy(ptweights) )

        #dsids = tree["DSID"].array(library="np")
        lund_zs = ak.flatten( tree["jetLundZ"].array(library="ak") )
        lund_kts =  ak.flatten( tree["jetLundKt"].array(library="ak") )
        lund_drs = ak.flatten( tree["jetLundDeltaR"].array(library="ak") )
        N_tracks = ak.flatten( tree["LRJ_Nconst_Charged"].array(library="ak") )
        MC_weights = ak.flatten( tree["LRJ_Nconst_Charged"].array(library="ak") )
        
        # ak.flatten(array, axis=None)
        lund_zs = ak.flatten(lund_zs, axis=None)
        lund_kts = ak.flatten(lund_kts, axis=None)
        lund_drs = ak.flatten(lund_drs, axis=None)
        N_tracks = ak.flatten(N_tracks, axis=None)
        #ak.to_numpy(ak_array)
        
        all_lund_zs = np.append(all_lund_zs, ak.to_numpy(lund_zs))
        all_lund_kts = np.append(all_lund_kts, ak.to_numpy(lund_kts) )
        all_lund_drs = np.append(all_lund_drs, ak.to_numpy(lund_drs) ) 
        all_lund_Ntrk = np.append(all_lund_Ntrk, ak.to_numpy(N_tracks) )

        print("files",files_count)
        print("total jets",len(all_lund_Ntrk))
        print("total selected jets",len(all_pt))
        print("total splittings",len(all_lund_kts))

        if files_count>150000: 
            break
#important thing, I just check and condition weight<150 just remove 2 jets inside Zprime test files. 

#print("removed points->", iii) 
hist_pt = np.histogram(all_pt, bins=100, range=(300,3000) )#, weights=all_weights)
fig, ax = plt.subplots()
print(hist_pt[1][:100],"  ",hist_pt[0])
ax.plot(hist_pt[1][:100], hist_pt[0])
ax.set_yscale("log", base=10)
#ax.ylim((0.001, 100000))
name_pt = "pt_Zprime_Noweights_R22"
#roc_auc = roc_auc_score(labels_test, nodes_out )
#ax.annotate(f'Area Under Curve {roc_auc:.4f}' , xy=(310, 120), xycoords='axes points',
#            size=12, ha='right', va='top',
#            bbox=dict(boxstyle='round', fc='w'))
plt.xlabel("pT", size=14)
plt.ylabel("# jets ", size=14)
plt.savefig(name_pt+'pt_histogram.png')
plt.show()

print("data loaded!")

'''
with open(name_pt, 'wb') as f:
    np.save( hist_pt , np.shape(hist_pt) )
    #np.save(f, np.array([1, 3]))
print(a, b)
'''

z = all_lund_zs
k = all_lund_kts
d = all_lund_drs
Ntrk = all_lund_Ntrk

#print(Ntrk)

#print("min z->",np.min(z))
#print("max  z->",np.max(z))
#print("min k->",np.min(k))
#print("max  k->",np.max(k))
#print("min d->",np.min(d))
#print("max  d->",np.max(d))
#print("min Ntrk->",np.min(Ntrk))
#print("max  Ntrk->",np.max(Ntrk))

z = z + 1e-4
k = k + 1e-4
d = d + 1e-4

######################################## for Jad 5% files ########################################
'''
### 
without + 1e-5 and using:
k[k == 0] = 1e-10
d[d == 0] = 1e-10
z[z == 0] = 1e-10

mean z-> 2.526054267987376
std  z-> 5.238387853035014
mean k-> -10.155155996582252
std  k-> 15.427801915205748
mean d-> 11.5151877928352
std  d-> 13.666163743674412
mean Ntrk-> 35.80012907092822
std  Ntrk-> 11.740907468153635
'''

''' 
using:
z = z + 1e-4
k = k + 1e-4
d = d + 1e-4

### 1% bkg and 1% signal top R22
mean z-> 2.0778886185997316
std  z-> 1.4714192378460587
mean k-> -2.3919176643005926
std  k-> 2.97112274201009
mean d-> 3.8924797952050296
std  d-> 2.279933125432541
mean Ntrk-> 0.0015256588314984825
std  Ntrk-> 0.6858312795401139

### 0.5% bkg R22 FIXED splitting train/test
mean z-> 2.0568479032747313
std  z-> 1.4450598054504056
mean k-> -2.379904791478249
std  k-> 2.940813577366582
mean d-> 3.8597358364389427
std  d-> 2.2748462855901073
mean Ntrk-> 57.588158609500134
std  Ntrk-> 23.900100132781983
'''

''' 5% bkg R22 mass>40GeV
total jets 4457145
total selected jets 2380595
total splittings 156405926
mean z-> 2.05694875947102
std  z-> 1.4451703069992108
mean k-> -2.3803571135800454
std  k-> 2.9412694759828963
mean d-> 3.860254858613063
std  d-> 2.275180549917065
mean Ntrk-> 26.561081140505863
std  Ntrk-> 16.51325032372397
'''
mean_z, std_z = 2.0568479032747313, 1.4450598054504056 ~ 0.1%
mean_dr, std_dr = 3.8597358364389427, 2.2748462855901073 ~ 0.1%
mean_kt, std_kt = -2.379904791478249, 2.940813577366582 ~ 


mean z-> 2.056   2.056
std  z-> 1.445   1.445
mean k-> -2.380  -2.379
std  k-> 2.941   2.940
mean d-> 3.860   3.859
std  d-> 2.275   2.274


z = np.log(1/z)
k = np.log(k)
d = np.log(1/d)
      
print("------------------------------------AFTER---------------------------------------")
print("mean z->",np.mean(z))
print("std  z->",np.std(z))
print("mean k->",np.mean(k))
print("std  k->",np.std(k))
print("mean d->",np.mean(d))
print("std  d->",np.std(d))
print("mean Ntrk->",np.mean(Ntrk))
print("std  Ntrk->",np.std(Ntrk))

