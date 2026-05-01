# light-roast-test

Example scripts for running on the UChicago AF.

## How to run these examples

To run the eventloop example, use the `python 3.6` kernel.

To run the columnar example, you'll need a more recent version of python, and you probably want to work in a virtual environment.  Here are a few options for that:  

### Using pip

The simplest way to get a new environment up and running is with `pip`:

```
bash # in case you're not already in a bash shell
new_env = "my_new_virtual_env"
python -m venv ${new_env}
source ${new_env}/bin/activate
pip install ipykernel numpy dask_jobqueue parse
python -m ipykernel install --user --name=${new_env}
```

Then restart your jupyter server and you should see the new kernel available for use.  To add a module later, just open a terminal in jupyter:

```
bash # in case you're not already in a bash shell
new_env = "my_new_virtual_env"
source ${new_env}/bin/activate
pip install scikit-learn
```

No need to restart the jupyter kernel if you're just adding packages.


### Using pixi
In a Jupyter shell, set up a new kernel to use:

```
curl -fsSL https://pixi.sh/install.sh | bash
export PATH=$PATH:.pixi/bin/
pixi init
pixi add python=3.12
pixi add atlas-schema
pixi add ipykernel
pixi add pixi-kernel
pixi add numpy                  # this syntax should work in most cases, but maybe not all packages support this
pixi add pip                    # if you need pip
pip install dask_jobqueue parse # this should also work
python -m ipykernel install --user --name=light-roast-kernel
```

Then restart your Jupyter server.  On restart, you should see an option to use the `light-roast-kernel`, which is the one you want.  You may need to add some additional packages, which you can do from within a shell:

```
pixi shell
pip install parse               # this should also work
```

You may need to restart your Jupyter server (not just the kernel) again if you do that.

### Using ALRB

If you prefer to use ALRB, you can try something like the following, but it hasn't worked smoothly in the past:

```
export ATLAS_LOCAL_ROOT_BASE="/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase"
alias setupATLAS="source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh"
setupATLAS
lsetup "python 3.11.9-x86_64-el9"
python3 -m venv lr-kernel
source lr-kernel/bin/activate
pip3 install --upgrade pip
pip3 install atlas-schema dask_jobqueue parse ipykernel
python3 -m ipykernel install --user --name=lr-kernel
```

The tool referenced [here](https://github.com/matthewfeickert/cvmfs-venv) may help with getting the environment right, but I'm not sure that's the problem.

## Analysis scripts

The following scripts implement the photon ABCD background estimation and R' studies for the SUSY γ+MET analysis.

| Script | Description | Output | Output location |
|---|---|---|---|
| `abcd_utils.py` | Shared utility library imported by all other analysis scripts. Provides event and photon selection masks (preselection, SR/VR regions, Tight/LoosePrime ID, isolation), truth category masks (Real, EFP, JFP, Other), ABCD region filling, background yield calculations (R', ABCD estimate, weighted averages), and JSON serialization helpers. | N/A — library only | N/A |
| `run_abcd.py` | Processes ROOT picontuples into per-sample ABCD JSON result files, then prints background estimates (data-driven JFP, MC JFP, and signal yields) for SR, VR, and Preselection regions. Optionally runs per-sample closure tests. | JSON result files; background estimate tables printed to stdout; per-sample pull plots (PDF) | `ABCD_results_<tag>/` |
| `calc_rprime.py` | Computes R′ = N_TT·N_LL / (N_LT·N_TL) for JFP+Other background MC and the data/MC scale factor SF from pre-computed JSON files, separately for Run 2 and Run 3 and for multiple LoosePrime working points. Also prints a VR data-vs-prediction comparison and a per-sample diagnostic table. Produces a LaTeX summary table. | Plain-text tables and LaTeX table printed to stdout | stdout |
| `abcd_stat_table.py` | Reads pre-computed ABCD JSON result files and prints a table of ABCD background estimates (TT_est = TL×LT/LL) with absolute and relative statistical uncertainties, broken down by LoosePrime working point, run era, and analysis region (SR/VR sub-regions). | Formatted text table printed to stdout | stdout |
| `check_json.py` | Utility script that reads ABCD JSON result files for Run 2 data and prints the run numbers and event numbers of individual events populating the TL, LT, and LL sideband regions of SR-0L-mT-low, for manual event inspection. | Run/event number list printed to stdout | stdout |
| `compare_abcd_tags.py` | Compares ABCD yields between two result directories (e.g., two processing versions), showing MC yields by truth category and data counts for each ABCD bin side-by-side with ratios. Useful for validating that a reprocessing has not introduced unexpected changes. | Formatted comparison tables printed to stdout | stdout |
| `rprime_analysis.py` | Unified R′ analysis tool that reads picontuples directly and computes R′ as a function of photon pT, |η|, conversion status, physics process, and analysis region. Supports optional pT-dependence corrections (two methods), systematic uncertainty tables, and per-variable LaTeX table output. | PDF/PNG plots; optional `.tex` table files | `rprime_analysis_output/` |
| `make_abcd_plots.py` | Produces weighted stacked histograms of photon isolation variables (topoetcone40/20, ptcone20, normalized to pT) split by MC truth category for Tight, LoosePrime4, and Loose ID criteria in Preselection-0L and VR-0L-mT-mid. Also produces JFP-only comparison plots with ratio-to-Tight panels. | PDF and PNG plots | `abcd_iso_plots/` |
| `plot_jfp_composition.py` | Diagnostic stacked-histogram plots of JFP and Other photon backgrounds broken down by physics process (Znunu, Wtaunu, Wmunu, Wenu) across all ABCD quadrants and analysis regions. Also produces process-composition summary plots showing absolute yield and process fraction per region. | PDF and PNG plots | `jfp_composition_plots/` |
| `plot_jfp_iso_vs_pt.py` | Isolation distributions for JFP photons in photon pT slices (10–15, 15–25, 25–40, >40 GeV), overlaying Tight, LoosePrime4, and Loose ID working points on a log-y scale with a ratio-to-Tight panel below. Covers Znunu, Wtaunu, and inclusive background in the Preselection-0L region. | PDF and PNG plots | `jfp_iso_pt_plots/` |
| `plot_jfp_iso_region.py` | Isolation distributions for JFP photons by analysis region, overlaying Tight/LP4/Loose working points, inclusive in pT and process. Optionally overlays the Preselection-0L shape for region-to-region shape comparison, with the ratio panel showing target/Preselection instead of LP4/Tight. | PDF and PNG plots | `jfp_iso_region_plots/` |
| `plot_truth_composition.py` | Stacked bar plots showing the truth-category composition (Prompt/EFP/JFP/Other fractions) of Tight photons across analysis regions for the Znunu, Wtaunu, and νν γ processes, with total weighted yields annotated on each bar. | PDF and PNG plots | `truth_composition_plots/` |
| `plot_truth_iso.py` | Isolation distributions (and reco-pT/truth-pT ratio) for Tight photons in pT slices, overlaying all truth categories (Prompt, EFP, JFP, Other) normalized to unit area for shape comparison. Covers Znunu, Wtaunu, and νν γ processes in the Preselection-0L region for Run 2, Run 3, and combined. | PDF and PNG plots | `truth_iso_plots/` |
| `study_wtaunu_mc.py` | Compares standard ABCD background estimation to a modified approach where Wtaunu JFP+Other is taken from MC rather than estimated data-driven. Prints side-by-side predictions, ABCD estimates, and scale factors (data/prediction) for VR and SR regions, and also prints separate R′ values for all-MC and non-Wtaunu-only MC. | Formatted comparison tables printed to stdout | stdout |

