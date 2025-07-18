import argparse
import awkward
import os.path as osp
import os
import glob
import torch
import awkward as ak
import time
import uproot
import uproot3
import yaml
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataListLoader, DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel, EdgeConv, GATConv,PNAConv
from torch_geometric.data import Data
import scipy.sparse as ss
from datetime import datetime, timedelta
from torch_geometric.utils import degree
import os.path

from tools.GNN_model_weight.models import *
from tools.GNN_model_weight.utils  import *
from tools.GNN_model_weight.utils_newdata  import *

print("Libraries loaded!")

scale_factor = 14.475606  

def main():

    if torch.cuda.is_available():
        torch.set_default_device('cuda')
    else:
        torch.set_default_device('cpu')
    device = torch.get_default_device()
    
    parser = argparse.ArgumentParser(description="Prepare data for classifier input")
    add_arg = parser.add_argument
    add_arg("config", help="job configuration file")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)
    config_signal = load_yaml("configs_FourProng/config_signal.yaml") # TODO: make this an optional argument, but then the same file needs to be used in utils_newdata.py
    config_signal = config_signal[config_signal["signal"]]  
    
    # path_to_test_file = "/data/jjhong96/LundNet_Ntuple/FourProngV1/HSbbWW_Val/*.FourProngV1_ANALYSIS.root/*.root"
    path_to_test_file = "/data/jjhong96/LundNet_Ntuple/FourProngV1/SS4W_Val/*.FourProngV1_ANALYSIS.root/*.root"
    # path_to_test_file = "/data/jjhong96/LundNet_Ntuple/FourProngV1/QCD_Val/user.jjhong.mc23_13p6TeV.801170.e8514_s4159_r15224_p6646.four_prongV1_ANALYSIS.root/user.jjhong.45345262._000026.ANALYSIS.root"
    # path_to_outdir = "/data/jjhong96/LundNet_Test/QCD_SS4W_SingleMass/SS4W_450_550/"
    path_to_outdir = "/data/jjhong96/LundNet_Test/QCD_SS4W_SplitStudy/"

    model_dic = {
        "SingleMass_125":"/data/jjhong96/LundNet_Model/QCD_SS4W_SingleMass/SS4W_40_150/LundNet_R22_QCD_SS4W_SingleMass_40_150_e050_0.16998.pt",
        "SingleMass_300":"/data/jjhong96/LundNet_Model/QCD_SS4W_SingleMass/SS4W_250_350/LundNet_R22_QCD_SS4W_SingleMass_250_350_e049_0.08852.pt",
        "SingleMass_500":"/data/jjhong96/LundNet_Model/QCD_SS4W_SingleMass/SS4W_450_550/LundNet_R22_QCD_SS4W_SingleMass_450_550_e049_0.04549.pt",
        "DoubleMass_125_300":"/data/jjhong96/LundNet_Model/QCD_SS4W_MassWindow/SS4W_125_300/LundNet_R22_QCD_SS4W_MassWindow_125_300_e050_0.19437.pt",
        "DoubleMass_125_500":"/data/jjhong96/LundNet_Model/QCD_SS4W_MassWindow/SS4W_125_500/LundNet_R22_QCD_SS4W_MassWindow_125_500_e050_0.16858.pt",
        "DoubleMass_300_500":"/data/jjhong96/LundNet_Model/QCD_SS4W_MassWindow/SS4W_300_500/LundNet_R22_QCD_SS4W_MassWindow_300_500_e048_0.08870.pt",
        "DoubleMass_300_500":"/data/jjhong96/LundNet_Model/QCD_SS4W_MassWindow/SS4W_300_500/LundNet_R22_QCD_SS4W_MassWindow_300_500_e048_0.08870.pt",
        "FullMass_10p":"/data/jjhong96/LundNet_Model/QCD_SS4W_JetNStudy/SS4W_40_800_10percent/LundNet_R22_QCD_SS4W_JetN_40_800_e043_0.24211.pt",
        "FullMass_25p":"/data/jjhong96/LundNet_Model/QCD_SS4W_JetNStudy/SS4W_40_800_25percent/LundNet_R22_QCD_SS4W_JetN_40_800_e049_0.23335.pt",
        "FullMass_50p":"/data/jjhong96/LundNet_Model/QCD_SS4W_JetNStudy/SS4W_40_800_50percent/LundNet_R22_QCD_SS4W_JetN_40_800_e049_0.20934.pt",
        "FullMass_90p":"/data/jjhong96/LundNet_Model/QCD_SS4W_JetNStudy/SS4W_40_800_90percent/LundNet_R22_QCD_SS4W_JetN_40_800_e047_0.20603.pt",
    }
    model_score_dict = {}
    # path_to_combined_ckpt = "/data/jjhong96/LundNet_Model/QCD_SS4W_SingleMass/SS4W_40_150/LundNet_R22_QCD_SS4W_SingleMass_40_150_e050_0.16998.pt" # JJ: Test, QCD vs SS4W 
    # path_to_combined_ckpt = "/data/jjhong96/LundNet_Model/QCD_SS4W_SingleMass/SS4W_250_350/LundNet_R22_QCD_SS4W_SingleMass_250_350_e049_0.08852.pt" # JJ: Test, QCD vs SS4W 
    # path_to_combined_ckpt = "/data/jjhong96/LundNet_Model/QCD_SS4W_SingleMass/SS4W_450_550/LundNet_R22_QCD_SS4W_SingleMass_450_550_e049_0.04549.pt" # JJ: Test, QCD vs SS4W classification
    output_name = "LundNetScores"
    # output_tag = "HSbbWW_Val"
    output_tag = "SS4W_Val"
    # output_tag = "QCD_Val"
    choose_model = "LundNet"
    learning_rate = 0.0005
    batch_size = 2048
    scale_factor = 1    

    
    files = glob.glob(path_to_test_file)

    #print ("files:",files)
    intreename = "AnalysisTree"

    nentries_total = 0
    nentries_done = 0

    for file in files:
        nentries_total += uproot3.numentries(file, intreename)

    print("Evaluating on {} files with {} entries in total.".format(len(files), nentries_total))
    
    #Load tf keras model
    # jet_type = "Akt10RecoChargedJet" #track jets
    jet_type = "Akt10UFOJet" #UFO jets

    t_filestart = time.time()

    count_files = 0 
    graph_small_example = []
    primary_Lund_only_one_arr=[]
    
    for file in files:
        t_start = time.time()
        dataset = []
        print("Loading file",file)

        with uproot.open(file) as infile:
            tree = infile[intreename]

            count_files += 1
            dsids = tree["dsid"].array(library="np")
            
            truthlabel = ak.to_numpy(ak.flatten(tree["LRJ_truthLabel"].array(library="ak")) )
            mcweights = tree["mcEventWeight"].array(library="np") #mcEventWeight
            nprong_labels = ak.to_numpy(ak.flatten(tree["LRJ_nprong"].array(library="ak")) )
            nQuarks = ak.to_numpy(ak.flatten(tree["LRJ_CapturedQuarkCount"].array(library="ak")) )
            GhostWBosonCount = ak.to_numpy(ak.flatten(tree["LRJ_GhostWBosonCount"].array(library="ak")) )
            
            parent1 =  ak.flatten(tree["jetLundIDParent1"].array(library="ak")) 
            parent2 = ak.flatten(tree["jetLundIDParent2"].array(library="ak")) 
            jet_pts = ak.to_numpy(ak.flatten(tree["LRJ_pt"].array(library="ak")) )
            jet_etas = ak.to_numpy(ak.flatten(tree["LRJ_eta"].array(library="ak")) )
            jet_phis = ak.to_numpy(ak.flatten(tree["LRJ_phi"].array(library="ak")) )
            jet_ms =  ak.to_numpy(ak.flatten(tree["LRJ_mass"].array(library="ak")))
            all_lund_zs = ak.flatten(tree["jetLundZ"].array(library="ak")) 
            all_lund_kts = ak.flatten(tree["jetLundKt"].array(library="ak")) 
            all_lund_drs = ak.flatten(tree["jetLundDeltaR"].array(library="ak"))
            N_tracks = ak.to_numpy(ak.flatten(tree["LRJ_Nconst_Charged"].array(library="ak")) )
            
            tau_21_wta = ak.to_numpy(ak.flatten(tree["Tau21_wta"].array(library="ak")))
            tau_32_wta = ak.to_numpy(ak.flatten(tree["Tau32_wta"].array(library="ak")))
            tau_43_wta = ak.to_numpy(ak.flatten(tree["Tau43_wta"].array(library="ak")))
            tau_42_wta = ak.to_numpy(ak.flatten(tree["Tau42_wta"].array(library="ak")))

            GN2X_score_branch_names = ["GN2Xv01_pqcd", "GN2Xv01_phbb", "GN2Xv01_ptop", "GN2Xv01_phcc"]
            GN2X_score_attribute_names = ["GN2X_pqcd", "GN2X_phbb", "GN2X_ptop", "GN2X_phcc"]
            GN2X_scores = {
                attribute_name: ak.flatten(tree[branch_name].array(entry_stop=entry_stop, library="ak"))
                for attribute_name, branch_name in zip(GN2X_score_attribute_names, GN2X_score_branch_names)
                if branch_name in tree
            }
            
            kT_selection = -2.0

            # flat_weights = GetPtWeight_2( dsids, jet_pts, filename=config['data']['weights_file'], SF=config['data']['scale_factor'])
            # flat_weights = GetPtWeight(jet_pts, nprong_labels, dsids[0], config_signal, 5)
            flat_weights = np.ones_like(jet_pts) # JJ: For testing, skip the pt flattenning
            # print('flat_weights: ', flat_weights)
            passed_selection = []   # will be a boolean array, True if jet passes selection
            Good_jets=[]
            mcweights_out=[]

            dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file_test_OLD( dataset, graph_small_example , all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, flat_weights, nprong_labels ,N_tracks,jet_pts, jet_ms, kT_selection, mcweights_out, Good_jets)

            # dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file_test(
            # dataset, graph_small_example, all_lund_zs, all_lund_kts, all_lund_drs,
            # parent1, parent2, flat_weights, nprong_labels, 
            # N_tracks, jet_pts, jet_ms,
            # kT_selection,
            # primary_Lund_only_one_arr,
            # passed_selection,
            # config_signal[signal]["signal_jet_nprong_label"]
            # )
            print("Dataset before Good_jets: ", len(dataset))
            for i in reversed(range(len(dataset))):
                if dataset[i] is graph_small_example:
                    Good_jets[i] = 0
                    dataset.pop(i)
            #     if jet_graph.y == 4:
            #         jet_graph.y = 1.0 # FourProng = signal
            #     elif jet_graph.y == 2:
            #         jet_graph.y = 0.0 # TwoProng = Background
            print("Dataset after Good_jets: ", len(dataset))
                
        #######################################################################################################################
        s_evt = 0
        events = 100
        print("Dataset created!")
        delta_t_fileax = time.time() - t_start
        print("Created dataset in {:.4f} seconds.".format(delta_t_fileax))

        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        print ("dataset dataset size:", len(dataset))


        #EVALUATING
        #torch.save(model.state_dict(), path)

        if choose_model == "LundNet":
            model = LundNet()
            # model = LundNet_old()
        if choose_model == "GATNet":
            model = GATNet()
        if choose_model == "GINNet":
            model = GINNet()
        if choose_model == "EdgeGinNet":
            model = EdgeGinNet()
        if choose_model == "PNANet":
            model = PNANet()

        for model_dic_tag in model_dic:
            model.load_state_dict(torch.load(model_dic[model_dic_tag]))
        # model.load_state_dict(torch.load(path_to_combined_ckpt))

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Usually gpu 4 worked best, it had the most memory available
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
            #Predict scores
            y_pred = get_scores(test_loader, model, device)
            #print(y_pred)
            tagger_scores = y_pred[:,0]
            model_score_dict[model_dic_tag] = np.array(tagger_scores)
            del tagger_scores
        
        delta_t_pred = time.time() - t_filestart - delta_t_fileax
        print("Calculated predicitions in {:.4f} seconds,".format(delta_t_pred))
        
        #Save root files containing model scores
        filename = file.split("/")[-1]
        outfile_path = os.path.join(path_to_outdir, filename)

        # tagger_scores += [-1] * (len(dsids) - len(tagger_scores))
        # tagger_scores = np.array(tagger_scores)

        print('Good_jets len: ', len(Good_jets))
        # print('Good_jets: ', Good_jets)

        Good_jet_selection = (np.array(Good_jets) == 1) # List of boolean
        print('dsids len before padding: ', len(dsids))

        dsids_padded = np.pad(dsids, (0, len(Good_jet_selection) - len(dsids)), constant_values=dsids[0]) # Do the padding
        dsids_padded = dsids_padded[Good_jet_selection]
        print('dsids len after padding: ', len(dsids_padded))
        truthlabel = truthlabel[Good_jet_selection]
        mcweights_padded = np.pad(mcweights, (0, len(Good_jet_selection) - len(dsids)), constant_values=mcweights[0]) # Do the padding
        mcweights_padded = mcweights_padded[Good_jet_selection]
        nprong_labels = nprong_labels[Good_jet_selection]
        nQuarks = nQuarks[Good_jet_selection]
        GhostWBosonCount = GhostWBosonCount[Good_jet_selection]
        flat_weights = flat_weights[Good_jet_selection]
        jet_pts = jet_pts[Good_jet_selection]
        jet_etas = jet_etas[Good_jet_selection]
        jet_phis = jet_phis[Good_jet_selection]
        jet_ms = jet_ms[Good_jet_selection]
        tau_21_wta = tau_21_wta[Good_jet_selection]
        tau_32_wta = tau_32_wta[Good_jet_selection]
        tau_43_wta = tau_43_wta[Good_jet_selection]
        tau_42_wta = tau_42_wta[Good_jet_selection]

        # tagger_scores = np.pad(tagger_scores, (0, len(dsids) - len(tagger_scores)), 'constant', constant_values=(-1))
        print ("dsids_padded ",len(dsids_padded),
               "mcweights_padded ", len(mcweights_padded), 
               "tagger_scores ",len(model_score_dict["SingleMass_125"]),
               "truthlabel ", len(truthlabel),
               "jet_pts ",len(jet_pts))

        
        
        with uproot.recreate("{}{}_{}_score_{}.root".format(path_to_outdir, output_tag, str(dsids[0]),output_name)) as f:
            treename = "FlatSubstructureJetTree"
            #Declare branch data types
            # f[treename] = uproot3.newtree({"EventInfo_mcChannelNumber": "int32",
            #                               "EventInfo_mcEventWeight": "float32",
            #                               "fjet_truthlabel": "int32",
            #                               "fjet_nPronglabel": "int32",
            #                               "fjet_nQuarklabel": "int32",
            #                               "fjet_GhostWBosonCount": "int32",
            #                               "fjet_flat_weights": "float32",
            #                               "fjet_nnscore": "float32",        # which is why I didn't include them
            #                               "fjet_pt": "float32",
            #                               "fjet_eta": "float32",
            #                               "fjet_phi": "float32",
            #                               "fjet_m": "float32",
            #                               "fjet_tau21_wta": "float32",
            #                               "fjet_tau32_wta": "float32",
            #                               "fjet_tau43_wta": "float32",
            #                               "fjet_tau42_wta": "float32",
            #                                # "truthjet_pt" : "float32",
            #                                # "ungroomedtruthjet_pt" : "float32",
            #                                # "ungroomedtruthjet_m" : "float32",
            #                                # "ungroomedtruthjet_split12" : "float32",
            #                               })

       
            #Save branches
            tree_dict = {
                                "EventInfo_mcChannelNumber": dsids_padded,
                                "EventInfo_mcEventWeight": mcweights_padded,
                                "fjet_truthlabel": truthlabel,
                                "fjet_nPronglabel": nprong_labels,
                                "fjet_nQuarklabel": nQuarks,
                                "fjet_flat_weights": flat_weights,
                                # "fjet_nnscore": tagger_scores,
                                "fjet_pt": jet_pts,
                                "fjet_eta": jet_etas,
                                "fjet_phi": jet_phis,
                                "fjet_m": jet_ms,
                                "fjet_tau21_wta": tau_21_wta,
                                "fjet_tau32_wta": tau_32_wta,
                                "fjet_tau43_wta": tau_43_wta,
                                "fjet_tau42_wta": tau_42_wta,
                                # "truthjet_pt" : truth_jetpt,
                                # "ungroomedtruthjet_pt" : truth_ungroomedjet_pt,
                                # "ungroomedtruthjet_m" : truth_ungroomedjet_m,
                                # "ungroomedtruthjet_split12" : truth_ungroomedjet_split12,
                                }
            
            for tag, scores in model_score_dict.items():
                tree_dict["fjet_{}_LundNetScore".format(tag)] = scores
                
            f[treename] = tree_dict
            

        delta_t_save = time.time() - t_start - delta_t_fileax - delta_t_pred
        print("Saved data in {:.4f} seconds.".format(delta_t_save))

        #nentries = 0
        #Time statistics
        nentries_done += uproot3.numentries(file, intreename)
        time_per_entry = (time.time() - t_start)/nentries_done
        eta = time_per_entry * (nentries_total - nentries_done)

        print("Evaluated on {} out of {} events".format(nentries_done, nentries_total))
        print("Estimated time until completion: {}".format(str(timedelta(seconds=eta))))


    print("Total evaluation time: {:.4f} seconds.".format(time.time()-t_filestart))

if __name__ == "__main__":
    main()