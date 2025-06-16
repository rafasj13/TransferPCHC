import pybnesian as pbn
import pandas as pd 
import numpy as np
from utils.util_metrics import *
from utils.util_draw import draw_model
from utils.util_syntethic import *
import json
import natsort as ns

def run(models, source_size, idx_changes, test_size, parentdict, spath, doPCHC=(True,True), alpha=0.05, kcv=5, patience=3, init=25, finish=1075, jump=50, iters=5, js_points=10000,seeds=(13,15)):
    """Run the models with the given parameters.
    Parameters
    ----------
    models : str --> Path of models to run.
    source_size : int --> Size for the source data.
    idx_changes :list -> Indexes to select the degree of modification for the sources (0:0%, 1:5%, 2:10%, 3:20%, 4:30%).
    test_size : int --> Size of the test set.   
    iters : int --> Number of iterations to run.
    seeds : tuple --> Tuple of two integers, the first is the seed for the training set and the second is the seed for the source data.
    init : int --> Initial size of the training set.
    finish : int --> Final size of the training set.
    jump : int --> Jump size for the loop.
    js_points : int --> Number of points for the JS divergence.
    alpha : float --> Significance level for the test.
    kcv : int --> Number of folds for cross-validation.
    patience : int --> Patience for the cross-validation.
    """
    modelPC = ['pcotransfer', 'pcot']
    modelHC = ['hctransfer', 'hc']
    doPC, doHC = doPCHC
    if doPC and doHC:
        MODELS = modelPC + modelHC
    elif doPC: 
        MODELS = modelPC
    elif doHC:
        MODELS = modelHC
    

    for train_size in range(init,finish,jump):    
    
        syntheticX = loadNETs(models)
        grouped_models = syntheticX.groupnames
        for name, percentages in ns.natsorted(grouped_models.items()):
        
            percentagesout = ns.natsorted(percentages)
            targetname = percentagesout[0]
            if test_size is None:
                training_target = syntheticX.sample(targetname, train_size, seed=seeds[0])
                tpath = models.split("/")[0]
                test_target = pd.read_csv(f"{tpath}/{targetname}_test.csv")
            else:
                data = syntheticX.sample(targetname, train_size+test_size, seed=seeds[0])
                training_target = data[:train_size]
                test_target = data[train_size:]

            source_list = []
            reference = syntheticX.get_model(targetname)
            parents = parentdict[name]
            for i in idx_changes:
                sourcedat = syntheticX.sample(percentagesout[i], source_size, seed=i+seeds[1])
                sourcedat = sourcedat + np.random.normal(0, 1, sourcedat.shape)
                source_list.append(sourcedat)

            results = {}
            for mkey in MODELS:
                results[mkey] = {'logl':[], 'time':[], 'hmd':[], 'complexity':[], 'shd':[]}

            trainClass = TrainModels(training_target, source_list, js_points=js_points)
            print(f"\n\n#### {name} with training size {train_size} ####")
            for i in range(iters):
                path = f"{spath}/{name}_T{train_size}"
                if doPC:
                    pcotransfer, logl_pcotransfer, time_pcotransfer = TrainModels.measure_time(trainClass.RCoT_KDE, test_target, transfer=True, alpha=alpha)
                    hmd_pcotransfer, complexity_pcotransfer,shd_pcotransfer = trainClass.structure_metrics(pcotransfer, reference)
                    results[modelPC[0]] = append_results(results[modelPC[0]], logl_pcotransfer.tolist(), time_pcotransfer, hmd_pcotransfer, complexity_pcotransfer, shd_pcotransfer)
                    
                    pcot, logl_pcot, time_pcot = TrainModels.measure_time(trainClass.RCoT_KDE, test_target, transfer=False, alpha=alpha)
                    hmd_pcot, complexity_pcot, shd_pcot = trainClass.structure_metrics(pcot, reference)
                    results[modelPC[1]] = append_results(results[modelPC[1]], logl_pcot.tolist(), time_pcot, hmd_pcot, complexity_pcot, shd_pcot)
                    
                    save_models([pcotransfer, pcot], path, [f"{modelPC[0]}_{i}", f"{modelPC[1]}_{i}"])
                if doHC:
                    hctransfer, logl_hctransfer, time_hctransfer  = TrainModels.measure_time(trainClass.HC_KDE, test_target, transfer=True, kcv=kcv, patience=patience, max_indegree=parents)    
                    hmd_hctransfer, complexity_hctransfer, shd_hctransfer = trainClass.structure_metrics(hctransfer, reference)
                    results[modelHC[0]] = append_results(results[modelHC[0]], logl_hctransfer.tolist(), time_hctransfer, hmd_hctransfer, complexity_hctransfer, shd_hctransfer)
                    
                    hc, logl_hc, time_hc = TrainModels.measure_time(trainClass.HC_KDE, test_target, transfer=False, kcv=kcv, patience=patience, max_indegree=parents)
                    hmd_hc, complexity_hc, shd_hc = trainClass.structure_metrics(hc, reference)    
                    results[modelHC[1]] = append_results(results[modelHC[1]], logl_hc.tolist(), time_hc, hmd_hc, complexity_hc, shd_hc)
                    
                    save_models([hctransfer, hc], path, [f"{modelHC[0]}_{i}", f"{modelHC[1]}_{i}"])
                
                # Save results to an existing JSON file
                output_file = path + ".json"
                if os.path.exists(output_file):
                    
                    with open(output_file, 'r') as f:
                        prev_results = json.load(f)
                        # Append new results to previous ones

                        for mkey in results.keys():
                            if mkey not in prev_results:
                                prev_results[mkey] = {'logl':[], 'time':[], 'hmd':[], 'complexity':[], 'shd':[]}
                            for metric in results[mkey].keys():
                                prev_results[mkey][metric] = results[mkey][metric]        
                    results = prev_results
                    with open(output_file, 'w') as f:
                        json.dump(prev_results, f, indent=4)
                
                else:
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=4)
            
                
            for key, metric in results.items():
                loglarr = np.array(metric['logl']) ## list of size iter, with each element being a numpy array of size test
                logl_sum = np.sum(loglarr, axis=1)
                print(f"Model:{key}, Time: {np.mean(metric['time']):.2f}, SLogL: {np.mean(logl_sum):.2f}, HMD: {np.mean(metric['hmd']):.2f}, 'COMPLEXITY': {np.mean(metric['complexity']):.2f}, SHD: {np.mean(metric['shd']):.2f}")
        

if __name__ == "__main__":
    test_size = 1024
    source_size = 3000 
    
    ##  SYNTHETIC DATA
    parent_synthetic = {"Synthetic1":3, "Synthetic2":5, "Synthetic3":1, "Synthetic4":1}
    parent_blearn = {"magic-niab":9, "magic-irri":11} 
    mod_010p = [0,2] 
    mod_51020p = [1,2,3]   
    # Run both PC and HC models
    run("asynthetic_nets/*/*.csv", source_size, mod_010p, test_size, parent_synthetic, "exps/Risk-Slocal_3k_010p/results_asynthetic", init=25, finish=1075, jump=100, seeds=(13,15), iters=3)
    run("asynthetic_nets/*/*.csv", source_size, mod_51020p, test_size, parent_synthetic, "exps/Risk-Slocal_3k_51020p/results_asynthetic", init=25, finish=1075, jump=100, seeds=(13,15), iters=3)
    
    run("bnlearn_magic/*/*.csv", source_size, mod_010p, test_size, parent_blearn, "exps/Risk-Slocal_3k_010p/results_bnlearn", init=25, finish=1075, jump=100, seeds=(13,15), iters=3)
    run("bnlearn_magic/*/*.csv", source_size, mod_51020p, test_size, parent_blearn, "exps/Risk-Slocal_3k_51020p/results_bnlearn", init=25, finish=1075, jump=100, seeds=(13,15), iters=3)

    
    
    ## UCI ML DATA
    parent_hc = {"1":5, "2":3, "3":3, "4":3, "5":5}
    parent_pc = {"1":7, "2":3, "3":5, "4":5, "5":9}
    
    mod_010p = [0,2] 
    mod_51020p = [1,2,3]  
    doPC = (True, False)
    doHC = (False, True)
    
    run("ci_nets_HC/*/*.csv", source_size, mod_010p, None, parent_hc, "exps/Risk-Slocal_3k_010p/results_uciml", doHC, init=25, finish=1075, jump=100, seeds=(13,15), iters=3)
    run("ci_nets_HC/*/*.csv", source_size, mod_51020p, None, parent_hc, "exps/Risk-Slocal_3k_51020p/results_uciml", doHC, init=25, finish=1075, jump=100, seeds=(13,15), iters=3)
    
    run("ci_nets_PC/*/*.csv", source_size, mod_010p, None, parent_pc, "exps/Risk-Slocal_3k_010p/results_uciml", doPC, init=25, finish=1075, jump=100, seeds=(13,15), iters=3)
    run("ci_nets_PC/*/*.csv", source_size, mod_51020p, None, parent_pc, "exps/Risk-Slocal_3k_51020p/results_uciml", doPC, init=125, finish=1075, jump=100, seeds=(13,15), iters=3)
