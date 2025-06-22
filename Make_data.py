import argparse
import os
import glob
import time
from datetime import timedelta
import gc
from operator import itemgetter

import uproot
import awkward as ak
import numpy as np
import torch

from tools.GNN_model_weight.utils_newdata import load_yaml, GetPtWeight, create_train_dataset_fulld_new_Ntrk_pt_weight_file

print("Libraries loaded!")

def main():
    parser = argparse.ArgumentParser(description="Prepare data for classifier input")
    add_arg = parser.add_argument
    add_arg("config", help="job configuration file")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)
    config_signal = load_yaml("configs/config_signal.yaml") # TODO: make this an optional argument, but then the same file needs to be used in utils_newdata.py
    signal = config_signal["signal"]

    path_to_files = config["path_to_trainfiles"]
    files = glob.glob(path_to_files)[:config["n_files"]]

    intreename = "AnalysisTree"

    n_files = len(files)
    print(f"Processing {n_files} files")
    t_start = time.time()

    # Jet properties that will be loaded and saved in the output ROOT file
    # which will accompany the graphs file;
    # these properties have one numerical value per jet
    # If a property is not present in the input file, it will be skipped without an error
    jet_property_names = {      # keys are output branch names, values are input branch names
        "fjet_m":              "LRJ_mass",
        "fjet_pt":             "LRJ_pt",
        "fjet_eta":            "LRJ_eta",
        "fjet_phi":            "LRJ_phi",
        "fjet_truth_label":    "LRJ_truthLabel",
        "fjet_Nconst_Charged": "LRJ_Nconst_Charged", # LRJ_Ntrk500, LRJ_Nconst?
        "GN2X_pqcd":           "GN2Xv01_pqcd",
        "GN2X_phbb":           "GN2Xv01_phbb",
        "GN2X_ptop":           "GN2Xv01_ptop",
        "GN2X_phcc":           "GN2Xv01_phcc",
    }
    # TODO: change this to just use the same names in the output file (requires modifying plotting code as well)
    
    # Additional variables which require some manipulation before they can be saved
    # because they need to be calculated or they have one value per event rather than per jet
    additional_output_vars = [
        "labels",                    # 1 for signal 0 for background
        "fjet_weight_pt",            # weight which makes pT distribution flat
        "EventInfo_mcEventWeight",
        "EventInfo_mcChannelNumber", # dsid
    ]
    out_tree_dict = {branch_name: ak.Array([]) for branch_name in [*jet_property_names.keys(), *additional_output_vars]}

    # Calculate flat-pT weights, apply jet selection and kT cuts, and construct the graphs
    dataset = []
    primary_Lund_only_one_arr = []
    for file_number, file in enumerate(files, start=1):
        print(f"\nLoading file: {file_number}/{n_files}\n", file)

        with uproot.open(file) as infile:
            tree = infile[intreename]

            dsids = tree["dsid"].array(library="np")
            dsid_test = dsids[0]                                 # check the first DSID, they should all be the same
            if dsid_test in config_signal[signal]["skip_dsids"]: # don't lose time with jets that don't pass pt cut or wrong signal sample
                print("Skipping file with DSID", dsid_test)
                continue

            # Determine how many entries to load based on fraction
            total_events = tree.num_entries
            entries_to_load = int(total_events * config["event_fraction"])
            entry_stop = min(entries_to_load, total_events)
            print(f"Loading {entry_stop} entries from {total_events} total entries")

            # Load the data
            jet_properties = {
                jet_property: ak.flatten(tree[jet_property].array(entry_stop=entry_stop, library="ak"))
                for jet_property in [*jet_property_names.values(), "jetLundZ", "jetLundKt", "jetLundDeltaR", "jetLundIDParent1", "jetLundIDParent2"]
                if jet_property in tree
            }
            truth_labels_unflattened = tree["LRJ_truthLabel"].array(entry_stop=entry_stop, library="ak")
            numbers_of_jets_per_event = ak.num(truth_labels_unflattened)

            mcEventWeights = tree["mcEventWeight"].array(entry_stop=entry_stop, library="np")
            jet_properties["EventInfo_mcEventWeight"] = np.repeat(mcEventWeights, numbers_of_jets_per_event)       # expand out the array so it has same length as flattened array
            jet_properties["EventInfo_mcChannelNumber"] = np.repeat(dsids[:entry_stop], numbers_of_jets_per_event) # TODO: can I do this without numpy? expand out the array so it has same length as flattened array

            # Calculate flat-pT weights
            print("\nCalculating weights:")
            jet_properties["fjet_weight_pt"] = GetPtWeight(jet_properties["LRJ_pt"], jet_properties["LRJ_truthLabel"], dsid_test, 5)

            passed_selection = []   # will be a boolean array, True if jet passes selection

            # Construct the graphs, applying jet selection and kT cuts
            print("\nCreating PyTorch graphs:")
            dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file(
                dataset,
                *itemgetter("jetLundZ", "jetLundKt", "jetLundDeltaR", "jetLundIDParent1", "jetLundIDParent2")(jet_properties),
                *itemgetter("fjet_weight_pt", "LRJ_truthLabel", "EventInfo_mcChannelNumber", "LRJ_Nconst_Charged", "LRJ_pt", "LRJ_mass")(jet_properties),
                GN2X_scores={
                    key: jet_properties[jet_property_names[key]]
                    for key in ["GN2X_pqcd", "GN2X_phbb", "GN2X_ptop", "GN2X_phcc"]
                    if jet_property_names[key] in jet_properties},
                kT_selection=config["kT_cut"],
                primary_Lund_only_one_arr=primary_Lund_only_one_arr,
                passed_selection=passed_selection,
                signal_jet_truth_label=config_signal[signal]["signal_jet_truth_label"],
                signal_dsid=config_signal[signal]["dsid"],
                pt_range=config_signal[signal]["pt_range"],
                mass_range=config_signal[signal]["mass_range"],
                include_pt=config["include_pt"],
            )

            for jet_property_out, jet_propety_in in jet_property_names.items():
                if jet_propety_in in jet_properties:
                    out_tree_dict[jet_property_out] = ak.concatenate([out_tree_dict[jet_property_out], jet_properties[jet_propety_in][passed_selection]])
                else:
                    print(f"Warning: {jet_propety_in} not found in file {file}, skipping")
                    if jet_property_out in dict: del out_tree_dict[jet_property_out]
            for output_var in additional_output_vars:
                append_array = ak.Array([jet_graph.y for jet_graph in dataset]) if output_var=="labels" else jet_properties[output_var][passed_selection]
                out_tree_dict[output_var] = ak.concatenate([out_tree_dict[output_var], append_array])

            gc.collect()

    print("\nDataset created! len():", len(dataset))
    delta_t_fileax = timedelta(seconds=round(time.time() - t_start))
    print(f"Time taken (hh:mm:ss): {delta_t_fileax}")

    # Construct output file names
    out_file_name_graphs = config["out_file_name_graphs"]
    outfile_name_root = config["out_file_name_root"]
    filepath_placeholder_vals = dict(
        id = config["id"],
        kT_cut = config["kT_cut"],
        include_pt = "_with_pt" if config["include_pt"] else ""
    )
    out_dir = config["out_dir"].format(**filepath_placeholder_vals)
    os.makedirs(out_dir, exist_ok=True)

    # Save the testing dataset, if specified
    test_frac = config["test_frac"]
    if test_frac is not None:
        print("Splitting dataset into train and test sets")
        test_num = int(len(dataset) * test_frac)
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        dataset = [dataset[i] for i in indices]
        dataset_test = dataset[:test_num]
        dataset = dataset[test_num:]

        out_file_name_graphs_test = out_file_name_graphs.format(
            **filepath_placeholder_vals,
            test_frac = f"_{int(test_frac*100)}percent"
        )
        output_path_graphs_test = os.path.join(out_dir, out_file_name_graphs_test)
        torch.save(dataset_test, output_path_graphs_test)
        print("Test graphs saved to:", output_path_graphs_test)

        outfile_name_root_test = outfile_name_root.format(
            **filepath_placeholder_vals,
            test_frac = f"_{int(test_frac*100)}percent"
        )
        output_path_root_test = os.path.join(out_dir, outfile_name_root_test)
        out_tree_dict_test = {}
        for key in out_tree_dict:
            out_tree_dict_test[key] = out_tree_dict[key][indices][:test_num]
            out_tree_dict[key] = out_tree_dict[key][indices][test_num:]
        with uproot.recreate(output_path_root_test) as outfile:
            outfile["FlatSubstructureJetTree"] = out_tree_dict_test
        print("Test dataset written to ROOT file:", output_path_root_test)

    # Save the training dataset
    out_file_name_graphs = out_file_name_graphs.format(
        **filepath_placeholder_vals,
        test_frac = f"_{int((1-test_frac)*100)}percent" if test_frac is not None else "",
    )
    output_path_graphs = os.path.join(out_dir, out_file_name_graphs)

    torch.save(dataset, output_path_graphs)
    print("Training graphs saved to:", output_path_graphs)

    outfile_name_root = outfile_name_root.format(
        **filepath_placeholder_vals,
        test_frac = f"_{int((1-test_frac)*100)}percent" if test_frac is not None else "",
    )
    output_path_root = os.path.join(out_dir, outfile_name_root)
    with uproot.recreate(os.path.join(output_path_root)) as outfile:
        outfile["FlatSubstructureJetTree"] = out_tree_dict
    print("Training dataset written to ROOT file:", output_path_root)


if __name__ == "__main__":
    main()
