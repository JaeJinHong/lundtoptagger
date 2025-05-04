import argparse
import os
import glob
import time
from datetime import timedelta
import gc

import uproot
import awkward as ak
import numpy as np
import torch

from tools.GNN_model_weight.utils_newdata import load_yaml, GetPtWeight_2, create_train_dataset_fulld_new_Ntrk_pt_weight_file

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

    dataset = []
    primary_Lund_only_one_arr = []

    for file_number, file in enumerate(files, start=1):
        print("\nLoading file", file)

        with uproot.open(file) as infile:
            tree = infile[intreename]

            dsids = tree["dsid"].array(library="np")
            dsid_test = dsids[0]                                 # check the first DSID, they should all be the same
            if dsid_test in config_signal[signal]["skip_dsids"]: # don't lose time with jets that don't pass pt cut or wrong signal sample
                continue

            truth_labels_unflattened = tree["LRJ_truthLabel"].array(library="ak")
            truth_labels = ak.flatten(truth_labels_unflattened)

            numbers_of_jets_per_event = ak.num(truth_labels_unflattened)

            mcEventWeights = tree["mcEventWeight"].array(library="np")
            mcEventWeights = np.repeat(mcEventWeights, numbers_of_jets_per_event) # expand out the array so it has same length as flattened array
            dsids = np.repeat(dsids, numbers_of_jets_per_event)            # TODO: can I do this without numpy? expand out the array so it has same length as flattened array

            print(f"length dataset: {len(dataset)}, file number: {file_number}/{n_files}")
            parent1 = ak.flatten(tree["jetLundIDParent1"].array(library="ak"))
            parent2 = ak.flatten(tree["jetLundIDParent2"].array(library="ak"))
            jet_ms = ak.flatten(tree["LRJ_mass"].array(library="ak"))
            jet_pts = ak.flatten(tree["LRJ_pt"].array(library="ak"))
            all_lund_zs = ak.flatten(tree["jetLundZ"].array(library="ak"))
            all_lund_kts = ak.flatten(tree["jetLundKt"].array(library="ak"))
            all_lund_drs = ak.flatten(tree["jetLundDeltaR"].array(library="ak"))
            N_tracks = ak.flatten(tree["LRJ_Nconst_Charged"].array(library="ak"))
            # N_tracks = ak.flatten(tree["LRJ_Ntrk500"].array(library="ak"))
            # N_tracks = ak.flatten(tree["LRJ_Nconst"].array(library="ak"))

            print("Calculating weights:")
            flat_weights = GetPtWeight_2(truth_labels, jet_pts, 5)
            kT_selection = config["kT_cut"]

            print("Creating PyTorch graphs:")
            dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file(
                dataset, all_lund_zs, all_lund_kts, all_lund_drs,
                parent1, parent2, flat_weights, truth_labels, dsids, mcEventWeights,
                N_tracks, jet_pts, jet_ms, kT_selection,
                primary_Lund_only_one_arr,
                config_signal[signal]["signal_jet_truth_label"],
                include_pt=config["include_pt"]
            )

            gc.collect()

    print("\nDataset created! len():", len(dataset))
    delta_t_fileax = timedelta(seconds=round(time.time() - t_start))
    print(f"Time taken (hh:mm:ss): {delta_t_fileax}")

    out_file_name = config["out_file_name"]
    out_dir = config["out_dir"].format(
        kT_cut = kT_selection,
        include_pt = "_with_pt" if config["include_pt"] else ""
    )
    os.makedirs(out_dir, exist_ok=True)

    test_frac = config["test_frac"]
    if test_frac is not None:
        print("Splitting dataset into train and test sets")
        test_num = int(len(dataset) * test_frac)
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        dataset = [dataset[i] for i in indices]
        dataset_test = dataset[:test_num]
        dataset = dataset[test_num:]

        print(f"_{test_frac*100}percent")
        out_file_name_test = out_file_name.format(
            kT_cut = kT_selection,
            include_pt = "_with_pt" if config["include_pt"] else "",
            test_frac = f"_{int(test_frac*100)}percent"
        )
        output_path_graphs_test = os.path.join(out_dir, out_file_name_test)
        torch.save(dataset_test, output_path_graphs_test)
        print("Test dataset saved to:", output_path_graphs_test)

    out_file_name = out_file_name.format(
        kT_cut = kT_selection,
        include_pt = "_with_pt" if config["include_pt"] else "",
        test_frac = f"_{int(1-test_frac*100)}percent" if test_frac is not None else ""
    )
    output_path_graphs = os.path.join(out_dir, out_file_name)

    torch.save(dataset, output_path_graphs)
    print("Dataset saved to:", output_path_graphs)


if __name__ == "__main__":
    main()
