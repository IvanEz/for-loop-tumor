from cmath import sqrt
import json
import math
import matplotlib
import numpy as np
import nibabel as nib
import os
import multiprocessing
from functools import partial
import shutil
import subprocess
import shlex

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyparsing import col
import seaborn as sns
from matplotlib import colors
from matplotlib.colors import TwoSlopeNorm
from seaborn.palettes import color_palette
import pymia.evaluation.metric as metric
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.writer as w
import pickle5 as pickle


"""
THRESHOLDS ARE KEY HYPERPARAMETERS => AT WHICH TUMOR DENSITY SHOULD IT BECOME VISIBLE?
"""
wt_threshold = 0.2
tc_threshold = 0.6

simulations_folder = "/Dataset/"


n_processes = 48

"""
TUMOR NEEDS TO BE WARPED TO SRI AND SEGMENTATIONS NEED TO BE MORPHED
TUMORCORE = CET+NECROSIS
WT = EDEMA+TUMORCORE
"""


def warp_sim_to_pat_space(input_file, ref_file, trans_file):
    atlas_nib = nib.load(atlas_file)
    with np.load(input_file) as data:
        sim_data = data['data']
    nib.save(nib.Nifti1Image(sim_data.astype(np.float32), atlas_nib.affine,
             atlas_nib.header), input_file.replace(".npz", ".nii.gz"))

    antstf_call = ("/ants/bin/antsApplyTransforms -d 3 -i " + input_file.replace(".npz", ".nii.gz") +
                   " -r " + ref_file +
                   " -t ./transform0GenericAffine.mat" +
                   " -t [" + trans_file + "0GenericAffine.mat,1]" +
                   " -t " + trans_file + "1InverseWarp.nii.gz" +
                   " -n Linear -o " + input_file.replace(".npz", ".nii.gz"))
    subprocess.run(shlex.split(antstf_call),
                   stdout=subprocess.PIPE, shell=False)

def warp_sim_to_pat_space_point(input_file, ref_file, trans_file):
    atlas_nib = nib.load(atlas_file)
    print("..........................." + input_file)
    with open(input_file, "rb") as par:
        #TODO: interpolate with manual formulas (e.g. uth: 10x - 7)
        #TODO: rounding to 6 digits?
        #paramsarray = np.zeros(8)
        params = pickle.load(par)
        x = params['icx']
        y = params['icy']
        z = params['icz']
    
    head, tail = os.path.split(input_file)
    #path = os.path.join(dir0,dir1,dir2,fn)
    with open(head + '/best_params.csv', 'w') as f:
    
        #f.write("x,y,z,t,label,mass,volume,count")
        #f.write('\n')
        f.write("{} {} {} 0 1 1 1 1".format(int(x*129),int(y*129),int(z*129)))
        
    antstf_call = ("/ants/bin/antsApplyTransforms -d 3 -i " + input_file.replace(".pkl", ".csv") +
                   
                   #" -t ./transform0GenericAffine.mat" +
                   " -t [" + trans_file + "0GenericAffine.mat,1]" +
                   " -t " + trans_file + "1InverseWarp.nii.gz" +
                   " -o " + input_file.replace(".pkl", "_out.csv"))
    subprocess.run(shlex.split(antstf_call),
                   stdout=subprocess.PIPE, shell=False)


"""
HELPER FUNCTIONS FOR THE DB QUERY
"""


def calc_dice(syn_data, real_data):
    """calculate the dice coefficient of the two input data"""
    combined = syn_data + real_data
    intersection = np.count_nonzero(combined == 2)
    union = np.count_nonzero(syn_data) + np.count_nonzero(real_data)
    if union == 0:
        return 0
    return (2 * intersection) / union


def calc_l2(syn_data, real_data):
    """calculate the l2 norm of the two input data"""
    combined = syn_data - real_data
    # linalg.norm uses the Frobenius norm as default ord
    return np.linalg.norm(combined)

def calc_hausdorff(syn_data, real_data):
    metrics = [metric.HausdorffDistance(percentile=95, metric='HDRFDST95')]
    labels = {1: 'FG'}
    evaluator = eval_.SegmentationEvaluator(metrics, labels)
    evaluator.evaluate(syn_data, real_data, "lala")
    results = evaluator.results[0].value
    #print(evaluator.results[0].value)
    #print(writer.ConsoleWriter().write(evaluator.results))
    return results

def get_scores_for_pair(measure_func, tc, wt, sim):
    """
    Calculate the similarity score of the passed tumor pair
    """
    # load simulation data
    with np.load(simulations_folder + sim + "/Data_0001.npz") as data:
        sim_tumor = data['data']
    # threshold
    sim_wt = np.ones(sim_tumor.shape)
    sim_wt[sim_tumor < wt_threshold] = 0
    sim_tc = np.ones(sim_tumor.shape)
    sim_tc[sim_tumor < tc_threshold] = 0

    # calc and update dice scores and partners
    cur_wt = measure_func(sim_wt, wt)
    cur_tc = measure_func(sim_tc, tc)
    combined = cur_wt + cur_tc
    scores = {}
    scores[sim] = {
        'tc': cur_tc,
        'wt': cur_wt,
        'combined': combined
    }

    # Hack to exclude low density tumors
    #if sim_tumor.max() < 0.5:
    #    scores[sim] = {
    #        'tc': 0,
    #        'wt': 0,
    #        'combined': 0
    #    }

    return scores


def get_scores_for_pair_bern(measure_func, tc, wt, sim):
    """
    Calculate the similarity score of the passed tumor pair
    """
    # load simulation data
    with np.load(simulations_folder + sim + "/Data_0001.npz") as data:
        sim_tumor = data['data']

    
    diff = sim_tumor - wt_threshold
    alpha = 0.5 + 0.5 * np.sign(diff) * (1.0 - np.exp( -diff * diff ))
    cur_wt = np.sum(np.multiply(alpha,wt))

    diff = sim_tumor - tc_threshold
    alpha = 0.5 + 0.5 * np.sign(diff) * (1.0 - np.exp( -diff * diff ))
    cur_tc = np.sum(np.multiply(alpha,tc))


    combined = cur_wt + cur_tc
    scores = {}
    scores[sim] = {
        'tc': cur_tc,
        'wt': cur_wt,
        'combined': combined
    }

    # Hack to exclude low density tumors
    #if sim_tumor.max() < 0.5:
    #    scores[sim] = {
    #        'tc': 0,
    #        'wt': 0,
    #        'combined': 0
    #    }

    return scores


def get_scores_for_real_tumor_parallel(processes: int = 32, score: str = "dice", ranks: int = 5):
    """
    Calculate the best similarity measure score of the given tumor based on the given dataset and return tuple (scores, best_score)
    scores - dump of the individual scores
    best_score - info about the best combined score
    """
    simulations = os.listdir(simulations_folder)
    simulations = [sim for sim in simulations if not "_single_slice" in sim]
    simulations.sort(key=lambda f: int(f))
    # cap test set to 128 for development - delete later!
    #simulations = simulations[:128]

    print("Starting parallel loop for {} folders with {} processes".format(
        len(simulations), processes))

    #measure_func = calc_dice if score == "dice" else calc_l2

    #measure_func = calc_dice if score == "dice" score == "dice" else calc_l2
    if score=="dice":
        measure_func = calc_dice
    elif score=="hd":
        measure_func = calc_hausdorff
    else:
        measure_func = calc_l2   

    func = partial(get_scores_for_pair_bern, measure_func,
                   tc, wt)
    with multiprocessing.Pool(processes) as pool:
        results = pool.map_async(func, simulations)
        single_scores = results.get()
        scores = {k: v for d in single_scores for k, v in d.items()}
        
    print(len(scores))
    # find best
    best_key_combined = 0
    best_key_tc = 0
    best_key_wt = 0

    if score == "dice" or score=="bern":
        best_key_combined = sorted(
            scores.keys(), key=lambda k: scores[k]['combined'])[-3]
        best_key_tc = sorted(scores.keys(), key=lambda k: scores[k]['tc'])[-3]
        best_key_wt = sorted(scores.keys(), key=lambda k: scores[k]['wt'])[-3]
    else:
        best_key_combined = min(
            scores.keys(), key=lambda k: scores[k]['combined'])
        best_key_tc = min(scores.keys(), key=lambda k: scores[k]['tc'])
        best_key_wt = min(scores.keys(), key=lambda k: scores[k]['wt'])

    best_score = {
        'best_score_combined': scores[best_key_combined]['combined'],
        'sim_combined': best_key_combined,
        'best_score_tc': scores[best_key_tc]['tc'],
        'sim_tc': best_key_tc,
        'best_score_wt': scores[best_key_wt]['wt'],
        'sim_wt': best_key_wt,
    }

    return scores, best_score

# adapted from Marcel Rosier


def plot_best_match_input_dice(folder: str = "./plots/", num=0, base_dir: str = "./olddataset_nopoints/"):
    font = {'family': 'DejaVu Sans',
            'weight': 'regular',
            'size': 8}

    matplotlib.rc('font', **font)
    fontSize = 8
    tumor_ids = os.listdir(base_dir)
    tumor_ids.sort(key=lambda f: int(f[3:6]))
    t1c_scores = []
    flair_scores = []
    entries = []
    spectral_palette = sns.color_palette("viridis", as_cmap=True)

    for tumor_id in tumor_ids:
        with open(f"{base_dir}/{tumor_id}/best_dice.json") as json_file:
            data = json.load(json_file)
            entries.append(data)


     # sort
    entries.sort(key=lambda entry: entry['best_score_combined'])
    # entries = entries[::2]
    t1c_scores = [entry['best_score_tc'] for entry in entries]
    flair_scores = [entry['best_score_wt'] for entry in entries]
    combined_scores = [entry['best_score_combined'] for entry in entries]

    fig, axes = plt.subplots(3, 1, sharex=True)
    for ax in axes:
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

    x_labels = np.linspace(0, len(entries), len(entries))
    y_ticks = np.linspace(0, 1, 11)

    t1c_avg = sum(t1c_scores) / len(t1c_scores)
    print(t1c_avg)
    # t1c_plot = sns.barplot(
    #     ax=axes[0], x=x_labels, y=t1c_scores, color="#3070B3")
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    # red_green_pal = sns.diverging_palette(0, 150, l=50, n=32, as_cmap=True)
    # rg2 = sns.color_palette("RdYlGn_r", 5)
    colors = [spectral_palette(norm(c)) for c in t1c_scores]

    t1c_plot = sns.scatterplot(
        ax=axes[0], x=x_labels, y=t1c_scores, hue=t1c_scores, palette=spectral_palette, legend=False)
    t1c_plot.set_xlim(-0.5)
    t1c_plot.set_ylim(-0.01, 1)
    # t1c_plot.set_title("T1Gd")
    # t1c_plot.set_xlabel("tumors")
    t1c_plot.set_xticklabels([])
    t1c_plot.set_ylabel("T1Gd dice score", size=fontSize)
    t1c_plot.set_yticks(y_ticks)
    t1c_plot.axhline(t1c_avg, color="purple")  # "#5ba56e")

    colors = [spectral_palette(norm(c)) for c in flair_scores]
    flair_avg = sum(flair_scores) / len(flair_scores)
    print(flair_avg)
    flair_plot = sns.scatterplot(
        ax=axes[1], x=x_labels, y=flair_scores, hue=flair_scores, palette=colors, legend=False)
    flair_plot.set_xlim(-0.5)
    flair_plot.set_ylim(-0.01, 1)
    # flair_plot.set_title("FLAIR")
    # flair_plot.set_xlabel("tumors")
    flair_plot.set_xticklabels([])
    flair_plot.set_ylabel("FLAIR dice score", size=fontSize)
    flair_plot.set_yticks(y_ticks)
    flair_plot.axhline(flair_avg, color="purple")  # "#5ba56e")

    print(t1c_scores)
    print(flair_scores)
    # combined plot
    y_ticks = np.linspace(0, 2, 21)
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2)
    # combined_scores = [t + f for t, f in zip(t1c_scores, flair_scores)]
    colors = [spectral_palette(norm(c)) for c in combined_scores]
    combined_scores_avg = sum(combined_scores) / len(combined_scores)
    print(combined_scores_avg)
    # combined_scores_plot = sns.barplot(
    #     ax=axes[2], x=x_labels, y=combined_scores, palette=colors)
    combined_scores_plot = sns.scatterplot(
        ax=axes[2], x=x_labels, y=combined_scores, hue=combined_scores, palette=colors, legend=False)
    combined_scores_plot.set_xlim(-0.5)
    combined_scores_plot.set_ylim(-0.01, 2)
    # combined_scores_plot.set_title("Combined")
    # combined_scores_plot.set_xlabel("tumors")
    combined_scores_plot.set_xticklabels([])
    combined_scores_plot.set_ylabel("Combined dice score", size=fontSize)
    combined_scores_plot.set_xlabel("Tumors", size=fontSize)
    combined_scores_plot.set_yticks(y_ticks)

    for label in combined_scores_plot.get_yticklabels()[1:][::2]:
        label.set_visible(False)
    combined_scores_plot.axhline(
        combined_scores_avg, color="purple")  # "#5ba56e")
    plt.show()
    plt.savefig(folder + "test_dice_olddataset_" + str(wt_threshold) + "_" + str(tc_threshold) + "_" + str(num) +
                ".png", bbox_inches='tight', dpi=800)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#MAIN
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

ordner = "./data_folder/"
scans = os.listdir(ordner)
index = 1
total = len(scans)

for scan in scans:
    t1c_file = ordner + scan + "/t1.nii.gz"
    seg_t1_file = ordner + scan + "/T1_binarized.nii.gz"
    seg_flair_file = ordner + scan + "/Flair.nii.gz"

    # Registrations
    
    antsSyN_call = "antsRegistrationSyN.sh -d 3 -n 32 -t s -f " + atlas_file + " -m " + t1c_file.replace(
        ".nii.gz", "space.nii.gz") + " -x " + atlas_mask_file + " -o " + t1c_file.replace(".nii.gz", "Reg.nii.gz")
    subprocess.run(shlex.split(antsSyN_call),
                   stdout=subprocess.PIPE, shell=False)

  

    tc = nib.load(seg_t1_file.replace(
        ".nii.gz", "space.nii.gz")).get_fdata()
    wt = nib.load(seg_flair_file.replace(
        ".nii.gz", "space.nii.gz")).get_fdata()


    # DBquery
    scores, best_score = get_scores_for_real_tumor_parallel(
        processes=n_processes, score="dice", ranks=5)

    with open(ordner + scan + "/best_dice.json", "w") as file:
        json.dump(best_score, file)


    # Warp best sim back into patient space
    shutil.copy2(simulations_folder + best_score["sim_combined"] +
                 "/Data_0001.npz", ordner + scan + "/best_sim_combined.npz")
    shutil.copy2(simulations_folder + best_score["sim_combined"] +
                 "/supro_params.txt", ordner + scan + "/best_params_combined.txt")

    shutil.copy2(simulations_folder +
                 best_score["sim_tc"] + "/Data_0001.npz", ordner + scan + "/best_sim_tc.npz")
    shutil.copy2(simulations_folder + best_score["sim_tc"] +
                 "/supro_params.txt", ordner + scan + "/best_params_tc.txt")

    shutil.copy2(simulations_folder +
                 best_score["sim_wt"] + "/Data_0001.npz", ordner + scan + "/best_sim_wt.npz")
    shutil.copy2(simulations_folder + best_score["sim_wt"] +
                 "/supro_params.txt", ordner + scan + "/best_params_wt.txt")
    shutil.copy2(simulations_folder +
                 best_score["sim_wt"] + "/parameter_tag.pkl", ordner + scan + "/best_params.pkl")

    warp_sim_to_pat_space(ordner + scan + "/best_sim_combined.npz",
                          t1c_file, t1c_file.replace(".nii.gz", "Reg.nii.gz"))
    warp_sim_to_pat_space(ordner + scan + "/best_sim_tc.npz",
                          t1c_file, t1c_file.replace(".nii.gz", "Reg.nii.gz"))
    warp_sim_to_pat_space(ordner + scan + "/best_sim_wt.npz",
                          t1c_file, t1c_file.replace(".nii.gz", "Reg.nii.gz"))
    #warp_sim_to_pat_space_point(ordner + scan + "/best_params.pkl",
    #                      t1c_file, t1c_file.replace(".nii.gz", "Reg.nii.gz"))

    print("[ scan " + str(index) + " out of " + str(total) + " ] " + scan + ": ")
    print(best_score)

    # remove unneccessary files
    os.remove(ordner + scan + "/t1Reg.nii.gz1Warp.nii.gz")
    os.remove(ordner + scan + "/t1-space.nii.gz")
    os.remove(ordner + scan + "/Flair-space.nii.gz")
    os.remove(ordner + scan + "/T1_binarized-space.nii.gz")
    os.remove(ordner + scan + "/t1Reg.nii.gz0GenericAffine.mat")
    os.remove(ordner + scan + "/t1Reg.nii.gzInverseWarped.nii.gz")
    os.remove(ordner + scan + "/best_sim_combined.npz")
    os.remove(ordner + scan + "/best_sim_tc.npz")
    os.remove(ordner + scan + "/best_sim_wt.npz")
    os.remove(ordner + scan + "/t1Reg.nii.gz1InverseWarp.nii.gz")
    os.remove(ordner + scan + "/t1Reg.nii.gzWarped.nii.gz")

    index += 1

#plot_best_match_input_dice(folder="./plots/", num=4, base_dir=ordner)
