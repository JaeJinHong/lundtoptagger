#!/usr/bin/env python

from utils_plots import *
#from utils_plotsM2 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
import ROOT
from root_numpy import fill_hist  as fh
import warnings
warnings.filterwarnings('ignore')
import os



for variation in range(1):
    taggers = {}
    tagger_files = {}
    other_MC_tagger_files = {}
    if variation==0:
        outdir='./out/'
        
        ## FTAG1 TESTS
        #tagger_files["LundNet_class"]    = './FTAG1/LundNet_R22_None_ln_kT_Cut_LRJ_toptagging_FTAG_qcd005_top02.root'

        #tagger_files["LundNet"]    = './FTAG1/LundNet_R22_None_ln_kT_Cut_LRJ_toptagging_FTAG_qcd005_top02.root' ## just to compare with JETM2

        
        tagger_files["LundNet"]    = './FTAG1/LundNet_PLUS_R22_None_ln_kT_Cut_LRJ_toptagging_FTAG_qcd005_top02_try2.root'
        #tagger_files["LundNet"]    = './FTAG1/LundNetPLUS_R22_LRJ_qcd05_top10_b-tagging_6Wcontained_only_top.root'
        
    
        #tagger_files["LundNet_class"]    = './top_moredata/LundNet_R22_15P8QCD_75P8TOP_BS4800_LR0004MoreDataTestPythia.root'

        tagger_files["LundNet_class"]    = './top_moredata/LundNet_R22_-1_ln_kT_Cut_LRJ_NewNtrk_qcd75_top15_lr0003_modif_opt11_6000Pythia.root' ## LundNet

            


    try:
        os.system("mkdir {}".format(outdir))
    except ImportError:
        print("{} already exists".format(prefix))
    pass

    working_point = 0.5
    for t in tagger_files:
        print("init",t)
        taggers[t] = tagger_scores(t,tagger_files[t], working_point)
        #print("sdfghjkjhgfdfghgfgh")

    
    do_gn2x = True
    if do_gn2x:
        get_wp_tag_gn2x(taggers["LundNet"], working_point, prefix="GN2X")

        
    ##### cuts per pT bin using pythia sample
    pol_func = get_wp_tag_pol_func(taggers["LundNet_class"], working_point)
    #pol_func = get_wp_tag_pol_func(taggers["LundNet"], working_point) ## ANN

    
    for t in taggers:
        if taggers[t].name == "3var":
            continue
        if taggers[t].name == "HerwigAngular" or taggers[t].name == "HerwigDipole" or taggers[t].name == "SherpaCluster"  or taggers[t].name == "SherpaLund" : 
            get_tag_other_MC(taggers[t], pol_func, working_point)
            #get_tag_other_MC(taggers[t], pol_func, working_point)
        else:
            get_wp_tag(taggers[t],working_point, prefix=outdir)  ## smooth function
    
    
    
    gn2x_top_discriminant(taggers["LundNet"],prefix=outdir)
    
    make_rocs(taggers,prefix=outdir)
    # # ## Make plot vs mu
    # bgrej_mu(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point) ## fixed
    # bgrej_npv(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point) ## fixed
    # pt_spectrum(taggers,weight="fjet_weight_pt", prefix=outdir)
    # pt_sigeff(taggers,weight="fjet_weight_pt", prefix=outdir)
    # # ## Make plot background rejection vs pT
    make_efficiencies_all(taggers, prefix=outdir) ## fixed
    # pt_bgrej_all(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point) ## fixed
    # make_efficiencies_pt_all(taggers,  200,  500, weight="fjet_weight_pt", prefix=outdir, cutmass=False) ## fixed
    # make_efficiencies_pt_all(taggers,  500, 1000, weight="fjet_weight_pt", prefix=outdir, cutmass=False) ## fixed
    # make_efficiencies_pt_all(taggers, 1000, 2000, weight="fjet_weight_pt", prefix=outdir, cutmass=False) ## fixed
    # make_efficiencies_pt_all(taggers, 2000, 3000, weight="fjet_weight_pt", prefix=outdir, cutmass=False) ## fixed

    pt_bgrej(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point) ## fixed
    #pt_bgrej_prymary(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point)
    
    # pt_bgrej_mass(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point) ## fixed
    # # # # # #

    #####################---------------------------################################################
    '''
    pt_bgrej_otherMC(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point)
    plotAlternative(taggers, weight="fjet_weight_pt", prefix=outdir, NNorANN='NN', wp=working_point)
    pt_bgrej_otherMC_2(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point)
    pt_signal_eff_otherMC_2(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point)
    #'''
    pt_signal_eff_otherMC_2(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point)

    # # # # # # ## Make mass sculpting plots (inclusive and in bins of pT)
    mass_sculpting(taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point)  ## fixed
    mass_sculpting_ptcut(taggers, 300,  650, weight="fjet_weight_pt", prefix=outdir, wp=working_point)  ## fixed
    mass_sculpting_ptcut(taggers, 650, 1000, weight="fjet_weight_pt", prefix=outdir, wp=working_point)  ## fixed
    mass_sculpting_ptcut(taggers, 1000, 3000, weight="fjet_weight_pt", prefix=outdir, wp=working_point)  ## fixed


    # # # for t in taggers:
    # #     mass_sculpting(taggers[t], t, weight="fjet_weight_pt", prefix=outdir)  ## fixed
    # #     mass_sculpting_ptcut(taggers[t], t,  200,  500, weight="fjet_weight_pt", prefix=outdir)  ## fixed
    # #     mass_sculpting_ptcut(taggers[t], t,  500, 1000, weight="fjet_weight_pt", prefix=outdir)  ## fixed
    # #     mass_sculpting_ptcut(taggers[t], t, 1000, 3000, weight="fjet_weight_pt", prefix=outdir)  ## fixed
    #
    # ## Make background rejection vs signal efficiency plots (inclusive and in bins of pT)
    make_efficiencies_3var(taggers, prefix=outdir) ## fixed
    # make_efficiencies_3var_massCut(taggers, prefix=outdir) ## fixed
    # make_efficiencies_pt(taggers,  200,  500, weight="fjet_weight_pt", prefix=outdir, cutmass=False) ## fixed
    # make_efficiencies_pt(taggers,  500, 1000, weight="fjet_weight_pt", prefix=outdir, cutmass=False) ## fixed
    # make_efficiencies_pt(taggers, 1000, 2000, weight="fjet_weight_pt", prefix=outdir, cutmass=False) ## fixed
    # make_efficiencies_pt(taggers, 2000, 3000, weight="fjet_weight_pt", prefix=outdir, cutmass=False) ## fixed
    # make_efficiencies_pt(taggers,  200,  500, weight="fjet_weight_pt", prefix=outdir, cutmass=True) ## fixed
    # make_efficiencies_pt(taggers,  500, 1000, weight="fjet_weight_pt", prefix=outdir, cutmass=True) ## fixed
    # make_efficiencies_pt(taggers, 1000, 2000, weight="fjet_weight_pt", prefix=outdir, cutmass=True) ## fixed
    # make_efficiencies_pt(taggers, 2000, 3000, weight="fjet_weight_pt", prefix=outdir, cutmass=True) ## fixed





