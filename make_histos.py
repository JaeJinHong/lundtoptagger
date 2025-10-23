import argparse
import os
import glob
import time
from datetime import timedelta

import numpy as np
import uproot
import awkward as ak
from ROOT import TH1D, TFile

from tools.GNN_model_weight.utils_newdata import load_yaml

print("Libraries loaded!")

# parameters
config_signal_path = "configs_FourProng/config_signal.yaml"
signal = "Hbb"
# infiles_paths = ["/eos/home-t/tmlinare/Lund/jetetmiss/JETMDataMC/jpierre/run/submitDir-2025-01-06-1052-fee0 files 1-50/data-ANALYSIS/mc20_13TeV.802017.Py8EG_A14NNPDF23LO_WprimeWZ_flatpT_wideWmass.deriv.DAOD_JETM2.e8482_s3797_r13145_p5548.root"]
# infiles_paths = ["/eos/home-t/tmlinare/Lund/jetetmiss/JETMDataMC/jpierre/run/submitDir-2025-01-12-1919-bc4c W files 1-50 (not flat mass)/data-ANALYSIS/mc20_13TeV.801859.Py8EG_A14NNPDF23LO_WprimeWZ_flatpT.deriv.DAOD_JETM2.e8482_s3681_r13145_p5548.root"]
# infiles_paths = ["/data/jjhong96/LundNet_Ntuple/FourProngV2/SS4W_Train/*/*.ANALYSIS.root"]
# infiles_paths = ["/data/jjhong96/LundNet_Ntuple/FourProngV2/TTbar_Train/*/*.ANALYSIS.root"]
# infiles_paths = ["/data/jjhong96/LundNet_Ntuple/FourProngV2/WZ_Train/*/*.ANALYSIS.root"]
infiles_paths = ["/data/jjhong96/LundNet_Ntuple/FourProngV2/Hbb_Train/*/*.ANALYSIS.root"]
# infiles_paths = ["/data/jjhong96/LundNet_Ntuple/FourProngV2/QCD_Train/*/*.ANALYSIS.root"]
# outfile_path = "histos/mass_40-300/WBSMP8_flat.root"
# outfile_path = "histos/mass_40-300/WBSMP8.root"
# outfile_path = "histos/mass_40-300_pt_200-3100/qcdP8.root"
# outfile_path = "histos/mass_40-300_pt_200-3100/qcdSL.root"
# outfile_path = "histos/mass_40-300_pt_200-3100/qcdSC.root"
# outfile_path = "histos/mass_40-300_pt_200-3100/qcdHD.root"
# outfile_path = "histos/FourProngV2/SS4W.root"
# outfile_path = "histos/FourProngV2/TTbar.root"
# outfile_path = "histos/FourProngV2/WZ.root"
outfile_path = "histos/FourProngV2/Hbb.root"
# outfile_path = "histos/FourProngV2/QCD.root"
nbins = 100
# TODO: put these sets of parameters in a dictionary or config file;
# then no global declaration needed
# and it should also be easy to choose a whole set of parameters


def main():
    global config_signal_path, signal, infiles_paths, outfile_path, nbins

    parser = argparse.ArgumentParser(description="Prepare data for classifier input")
    add_arg = parser.add_argument
    add_arg("config",    default=config_signal_path, nargs="?", help="signal configuration file")
    add_arg("--signal",  default=signal,             help="Signal type (W or top), used to choose the configuration from the signal config file")
    add_arg("--infile",  default=infiles_paths,      nargs="+", help="Input file path(s) as glob pattern(s), separated by spaces.")
    add_arg("--outfile", default=outfile_path,       help="Output file path")
    add_arg("--nbins",   default=nbins,    type=int, help="Number of bins for histograms")
    add_arg("--pt_max",  default=3100,   type=float, help="Maximum pT value for histograms, only used if value in config is .inf")
    add_arg("--m_max",   default=3500,   type=float, help="Maximum mass value for histograms, only used if value in config is .inf")
    add_arg("--eta_max", default=5.0, type=float, help="Maximum pseudorapidity value for histograms, only used if value in config is .inf")
    args = parser.parse_args()

    config_signal_path = args.config
    signal = args.signal
    infile_patterns = args.infile
    outfile_path = args.outfile
    nbins = args.nbins

    config_signal = load_yaml(config_signal_path)
    pt_min, pt_max = config_signal[signal]["pt_range"]
    m_min, m_max = config_signal[signal]["mass_range"]
    eta_max = config_signal[signal]["eta_max"]
    if pt_max == float("inf"):
        pt_max = args.pt_max
    if m_max == float("inf"):
        m_max = args.m_max
    if eta_max == float("inf"):
        eta_max = args.eta_max

    # Create list of files from glob patterns
    files = []
    for pattern in infile_patterns:
        pattern = pattern.strip()  # remove any accidental leading/trailing whitespace
        if pattern:
            files.extend(glob.glob(pattern))
    # Remove duplicates and sort for consistency
    files = sorted(set(files))

    intreename = "AnalysisTree"

    print(f"Processing {len(files)} files")
    t_start = time.time()

    # Create histograms
    hist_pt = TH1D("pt", "Jet pT histogram", nbins, pt_min, pt_max)
    hist_m = TH1D("mass", "Jet mass histogram", nbins, m_min, m_max)
    hist_eta = TH1D("eta", "Jet eta histogram", nbins, -eta_max, eta_max)
    hist_truth_label = TH1D("truth_label", "Jet truth label histogram", 13, -0.5, 12.5)
    hist_nprong_label = TH1D("nprong_label", "Jet n-prong truth label histogram", 5, -0.5, 4.5)
    hist_nquark10_label = TH1D("nquark10_label", "Jet n-quark (dR < 1.0) truth label histogram", 6, -0.5, 5.5)
    hist_nquark8_label = TH1D("nquark8_label", "Jet n-quark (dR < 0.8) truth label histogram", 6, -0.5, 5.5)
    hist_nquark8_loose_iso_label = TH1D("nquark8_loose_iso_label", "Jet n-quark (dR < 0.8, iso dR > 0.02) truth label histogram", 6, -0.5, 5.5)
    hist_nquark8_medium_iso_label = TH1D("nquark8_medium_iso_label", "Jet n-quark (dR < 0.8, iso dR > 0.05) truth label histogram", 6, -0.5, 5.5)
    hist_nquark8_tight_iso_label = TH1D("nquark8_tight_iso_label", "Jet n-quark (dR < 0.8, iso dR > 0.1) truth label histogram", 6, -0.5, 5.5)


    for file_number, file in enumerate(files, start=1):
        print(f"\nLoading file {file_number}/{len(files)}: {file}")

        with uproot.open(file) as infile:
            tree = infile[intreename]

            dsid_test = tree["dsid"].array(library="np")[0]
            jet_labels = config_signal[signal]["signal_jet_label"] if dsid_test in config_signal[signal]["dsids"] else 10
            truth_labels = ak.to_numpy(ak.flatten(tree["LRJ_truthLabel"].array()))
            nprong_labels = ak.to_numpy(ak.flatten(tree["LRJ_nprong"].array()))
            nquark10_labels = ak.to_numpy(ak.flatten(tree["LRJ_nquark10"].array()))
            nquark8_labels = ak.to_numpy(ak.flatten(tree["LRJ_nquark8"].array()))
            nquark8_loose_iso_labels = ak.to_numpy(ak.flatten(tree["LRJ_nquark8_loose_iso"].array()))
            nquark8_medium_iso_labels = ak.to_numpy(ak.flatten(tree["LRJ_nquark8_medium_iso"].array()))
            nquark8_tight_iso_labels = ak.to_numpy(ak.flatten(tree["LRJ_nquark8_tight_iso"].array()))
            jet_masses = ak.to_numpy(ak.flatten(tree["LRJ_mass"].array()))
            jet_pts = ak.to_numpy(ak.flatten(tree["LRJ_pt"].array()))
            jet_etas = ak.to_numpy(ak.flatten(tree["LRJ_eta"].array()))

            selection = np.isin(nprong_labels, jet_labels) \
                      & (jet_masses >= m_min) & (jet_masses <= m_max) \
                      & (jet_pts >= pt_min) & (jet_pts <= pt_max) \
                      & (abs(jet_etas) <= eta_max)
            
            jet_masses = jet_masses[selection]
            jet_pts = jet_pts[selection]
            jet_etas = jet_etas[selection]
            truth_labels = truth_labels[selection]
            nprong_labels = nprong_labels[selection]
            nquark10_labels = nquark10_labels[selection]
            nquark8_labels = nquark8_labels[selection]
            nquark8_loose_iso_labels = nquark8_loose_iso_labels[selection]
            nquark8_medium_iso_labels = nquark8_medium_iso_labels[selection]
            nquark8_tight_iso_labels = nquark8_tight_iso_labels[selection]

        # Fill histograms
        njets = len(jet_pts)
        if njets == 0:
            print("  No jets passing selection in this file.")
            continue
        weights = np.ones(njets)
        hist_pt.FillN(njets, jet_pts, weights)
        hist_m.FillN(njets, jet_masses, weights)
        hist_eta.FillN(njets, jet_etas, weights)
        hist_truth_label.FillN(njets, truth_labels.astype(float), weights)
        hist_nprong_label.FillN(njets, nprong_labels.astype(float), weights)
        hist_nquark10_label.FillN(njets, nquark10_labels.astype(float), weights)
        hist_nquark8_label.FillN(njets, nquark8_labels.astype(float), weights)
        hist_nquark8_loose_iso_label.FillN(njets, nquark8_loose_iso_labels.astype(float), weights)
        hist_nquark8_medium_iso_label.FillN(njets, nquark8_medium_iso_labels.astype(float), weights)
        hist_nquark8_tight_iso_label.FillN(njets, nquark8_tight_iso_labels.astype(float), weights)

    # Save histograms to a ROOT file
    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
    output_file = TFile(outfile_path, "RECREATE")
    hist_pt.Write()
    hist_m.Write()
    hist_eta.Write()
    hist_truth_label.Write()
    hist_nprong_label.Write()
    hist_nquark10_label.Write()
    hist_nquark8_label.Write()
    hist_nquark8_loose_iso_label.Write()
    hist_nquark8_medium_iso_label.Write()
    hist_nquark8_tight_iso_label.Write()
    output_file.Close()

    print(f"Histograms saved to {outfile_path}")

    delta_t_fileax = timedelta(seconds=round(time.time() - t_start))
    print(f"Time taken (hh:mm:ss): {delta_t_fileax}")


if __name__ == "__main__":
    main()