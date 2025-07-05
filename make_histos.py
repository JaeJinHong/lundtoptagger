import argparse
import os
import glob
import time
from datetime import timedelta

import numpy as np
import uproot
import awkward as ak
from ROOT import TH1F, TFile

from tools.GNN_model_weight.utils_newdata import load_yaml

print("Libraries loaded!")

# parameters
config_signal_path = "configs/config_signal.yaml"
signal = "W"
# infiles_paths = ["/eos/home-t/tmlinare/Lund/jetetmiss/JETMDataMC/jpierre/run/submitDir-2025-01-06-1052-fee0 files 1-50/data-ANALYSIS/mc20_13TeV.802017.Py8EG_A14NNPDF23LO_WprimeWZ_flatpT_wideWmass.deriv.DAOD_JETM2.e8482_s3797_r13145_p5548.root"]
# infiles_paths = ["/eos/home-t/tmlinare/Lund/jetetmiss/JETMDataMC/jpierre/run/submitDir-2025-01-12-1919-bc4c W files 1-50 (not flat mass)/data-ANALYSIS/mc20_13TeV.801859.Py8EG_A14NNPDF23LO_WprimeWZ_flatpT.deriv.DAOD_JETM2.e8482_s3681_r13145_p5548.root"]
infiles_paths = ["/eos/user/r/ravinasc/R_22_Samples/JETM2_mc20/Pythia_train/Pythia_qcd_01/*.root"]
# infiles_paths = ["/eos/user/r/ravinasc/R_22_Samples/JETM2_mc20/Alternative_MC/Sherpa_Lund/*.root"]
# infiles_paths = ["/eos/user/r/ravinasc/R_22_Samples/JETM2_mc20/Alternative_MC/Sherpa_Cluster/*.root"]
# infiles_paths = ["/eos/user/r/ravinasc/R_22_Samples/JETM2_mc20/Alternative_MC/Herwing_dipole/*.root"]
# outfile_path = "histos/mass_40-300/WBSMP8_flat.root"
# outfile_path = "histos/mass_40-300/WBSMP8.root"
outfile_path = "histos/mass_40-300_pt_200-3100/qcdP8.root"
# outfile_path = "histos/mass_40-300_pt_200-3100/qcdSL.root"
# outfile_path = "histos/mass_40-300_pt_200-3100/qcdSC.root"
# outfile_path = "histos/mass_40-300_pt_200-3100/qcdHD.root"
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
    args = parser.parse_args()

    config_signal_path = args.config
    signal = args.signal
    infile_patterns = args.infile
    outfile_path = args.outfile
    nbins = args.nbins

    config_signal = load_yaml(config_signal_path)
    pt_min, pt_max = config_signal[signal]["pt_range"]
    m_min, m_max = config_signal[signal]["mass_range"]
    if pt_max == float("inf"):
        pt_max = args.pt_max
    if m_max == float("inf"):
        m_max = args.m_max

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
    hist_pt = TH1F("pt", "Jet pT histogram", nbins, pt_min, pt_max)
    hist_m = TH1F("mass", "Jet mass histogram", nbins, m_min, m_max)
    hist_truth_label = TH1F("truth_label", "Jet truth label histogram", 11, -0.5, 10.5)

    for file_number, file in enumerate(files, start=1):
        print(f"\nLoading file {file_number}/{len(files)}: {file}")

        with uproot.open(file) as infile:
            tree = infile[intreename]

            dsid_test = tree["dsid"].array(library="np")[0]
            jet_truth_labels = config_signal[signal]["signal_jet_truth_labels"] if dsid_test in config_signal[signal]["dsids"] else 10
            truth_labels = ak.flatten(tree["LRJ_truthLabel"].array())
            jet_masses = ak.flatten(tree["LRJ_mass"].array())
            jet_pts = ak.flatten(tree["LRJ_pt"].array())

            selection = np.isin(ak.to_numpy(truth_labels), jet_truth_labels) \
                      & (jet_masses >= m_min) & (jet_masses <= m_max) \
                      & (jet_pts >= pt_min) & (jet_pts <= pt_max)
            jet_masses = jet_masses[selection]
            jet_pts = jet_pts[selection]
            truth_labels = truth_labels[selection]

        # Fill histograms
        for pt in jet_pts:
            hist_pt.Fill(pt)
        for m in jet_masses:
            hist_m.Fill(m)
        for label in truth_labels:
            hist_truth_label.Fill(label)

    # Save histograms to a ROOT file
    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
    output_file = TFile(outfile_path, "RECREATE")
    hist_pt.Write()
    hist_m.Write()
    hist_truth_label.Write()
    output_file.Close()

    print(f"Histograms saved to {outfile_path}")

    delta_t_fileax = timedelta(seconds=round(time.time() - t_start))
    print(f"Time taken (hh:mm:ss): {delta_t_fileax}")


if __name__ == "__main__":
    main()