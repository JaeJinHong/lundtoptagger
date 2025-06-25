import argparse
import os
import glob
import time
import gc

import uproot
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from tools.GNN_model_weight.models import *
from tools.GNN_model_weight.utils_newdata import load_yaml, get_scores

print("Libraries loaded!")

def main():
    parser = argparse.ArgumentParser(description='Train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    add_arg('--ln_kT_cut', type=float, help="minimum value of kT kept for the training graphs")
    add_arg('--sample', type=float, help="sample identifier which should be part of input and output file names")
    args = parser.parse_args()

    config_file = args.config
    config = load_yaml(config_file)

    kT_selection = args.ln_kT_cut if args.ln_kT_cut is not None else config['data']['kT_cut']
    filepath_placeholder_vals = dict(
        sample = args.sample if args.sample is not None else config['data']['sample'],
        kT_cut = kT_selection
    )

    path_to_test_file_root = config['data']['path_to_test_file_root'].format(**filepath_placeholder_vals)
    files_root = glob.glob(path_to_test_file_root)
    print ("path_to_test_file_root:", path_to_test_file_root)
    print ("files:", files_root)

    path_to_test_file_graphs = config['data']['path_to_test_file_graphs'].format(**filepath_placeholder_vals)
    files_graphs = glob.glob(path_to_test_file_graphs)
    print ("path_to_test_file_graphs:", path_to_test_file_graphs)
    print ("files:", files_graphs)

    path_to_outdir = config['data']['path_to_outdir'].format(**filepath_placeholder_vals)
    os.makedirs(path_to_outdir, exist_ok=True)
    print("The output files will be saved to")
    print(path_to_outdir)

    path_to_combined_ckpt = config['test']['path_to_combined_ckpt'][kT_selection]
    print("ckpt used:", path_to_combined_ckpt )

    output_name = config['test']['output_name'].format(**filepath_placeholder_vals)

    intreename = "FlatSubstructureJetTree"
    files_and_trees = {file_name: intreename for file_name in files_root}
    nentries_total = sum(entry[-1] for entry in uproot.num_entries(files_and_trees))
    nentries_done = 0

    batch_size = config['test']['batch_size']
    choose_model = config['test']['choose_model']

    t_filestart = time.time()

    # Set up model
    # TODO: test multiple models, so there is no need to re-load the data for each model
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
    if choose_model == "LundNet_plus_GN2X":
        model = LundNet_plus_GN2X()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Usually gpu 4 worked best, it had the most memory available
    model.load_state_dict(torch.load(path_to_combined_ckpt, map_location=device))
    print(f'\nUsing device: {device}')
    model.to(device)

    # Evaluation
    for file_number, (file_graphs, file_root) in enumerate(zip(files_graphs,files_root), start=1):
        t_start = time.time()

        # Load the data
        print(f"\nLoading file: {file_number}/{len(files_graphs)}\n", file_graphs)

        dataset = torch.load(file_graphs, weights_only=False)

        n_jets = len(dataset)
        print("Dataset size:", n_jets)
        delta_t_fileax = time.time() - t_start
        minutes, seconds = divmod(round(delta_t_fileax), 60)
        print(f"Time taken to load: {minutes:d} min {seconds:d} s")

        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Predict scores
        print("\nCalculating scores...")
        y_pred = get_scores(test_loader, model, device)
        tagger_scores = np.array(y_pred[:,0])

        delta_t_pred = time.time() - t_start - delta_t_fileax
        minutes, seconds = divmod(round(delta_t_pred), 60)
        print(f"Time taken to calculate predictions: {minutes:d} min {seconds:d} s")

        # Free up memory
        del dataset, test_loader, y_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get the tree from the input ROOT file and add the scores to it
        print("\nSaving scores to ROOT file...")
        with uproot.open(file_root) as f:
            arrays = f[intreename].arrays()
        arrays[config["test"]["scores_branch_name"].format(model=choose_model)] = tagger_scores

        # Save ROOT files containing model scores
        # TODO: just add scores to existing ROOT files instead of creating new ones; can keep adding scores for different models
        # (this can already be done by setting the output path equal to the input path, but is inefficient,
        # since the whole tree is read and written again, with PyROOT it is possible to add new branches to an existing tree)
        filename_no_ext = os.path.splitext(os.path.basename(file_root))[0]  # get the input file name without the .root extension
        outfile_path = os.path.join(path_to_outdir, filename_no_ext) + f"{output_name}.root"
        outfile_path = outfile_path.format(**filepath_placeholder_vals)

        with uproot.recreate(outfile_path) as f:
            f["FlatSubstructureJetTree"] = arrays
        print("Scores saved to:", outfile_path)

        delta_t_save = time.time() - t_start - delta_t_fileax - delta_t_pred
        minutes, seconds = divmod(round(delta_t_save), 60)
        print(f"Time taken to save: {minutes:d} min {seconds:d} s")

        # Time statistics
        nentries_done += n_jets
        time_per_entry = (time.time() - t_start)/(nentries_done)
        eta = time_per_entry * (nentries_total - nentries_done)
        minutes, seconds = divmod(round(eta), 60)

        # Free up memory
        del arrays, tagger_scores
        gc.collect()

        print(f"\nEvaluated on {nentries_done} out of {nentries_total} jets")
        print(f"Estimated time until completion: {minutes:d} min {seconds:d} s")

    delta_t_total = time.time()-t_filestart
    minutes, seconds = divmod(round(delta_t_total), 60)
    print(f"\nTotal evaluation time: {minutes:d} min {seconds:d} s")


if __name__ == "__main__":
    main()
