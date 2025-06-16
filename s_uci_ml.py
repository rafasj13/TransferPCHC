import pybnesian as pbn
import pandas as pd 
import numpy as np
from utils.util_syntethic import *
import random
import os
import glob
from utils.util_metrics import hamming_distance
from utils.util_draw import draw_model
  

def modify_arcs_and_sample(path, trainsample, testsample, sample_size, percentages, output_dir, seed=15, **kwargs):
    np.random.seed(seed)
    random.seed(seed)
    basename = os.path.basename(path)[:-4]
    spath = f"{output_dir}/{basename}"
    os.makedirs(spath, exist_ok=True)
    
    
    netHC_orig, netPC_orig, sourcedat = train_reference(path, trainsample, testsample, spath, seed=seed, **kwargs)
    netHC_orig.save(f"{output_dir}/{basename}_HC")
    netPC_orig.save(f"{output_dir}/{basename}_PC")
    
    
    for prefix, net_orig in zip(['HC','PC'],[netHC_orig, netPC_orig]):
        print(f"\n### SAMPLE {prefix} model for {basename} ###")
        # 2. Save arcs sample from orginal network
        sample_s = net_orig.sample(sample_size).to_pandas()
        orig_arcs = net_orig.arcs()
        arcs_s = pd.DataFrame(orig_arcs, columns=['from', 'to'])
        sample_s.to_csv(spath + f"/{basename}_sampled.csv", index=False)
        arcs_s.to_csv(spath + f"/{prefix}_{basename}_arcs.csv", index=False)
        
        
        # 2. Remove random arcs
        nodes = net_orig.nodes()
        n_arcs = len(orig_arcs)
        for percent_to_modify in percentages:
            print(f"\n### {basename} with {percent_to_modify*100}% modified ###")
            n_modify = int(np.ceil(n_arcs * percent_to_modify))
            arcs_to_remove = random.sample(orig_arcs, n_modify)
            mod_arcs = [arc for arc in orig_arcs if arc not in arcs_to_remove]
            
            # 3. Add new arcs (without creating cycles)
            possible_arcs = [(from_, to) for from_ in nodes for to in nodes if from_ != to and (from_, to) not in orig_arcs]
            added = 0
            tries = 0
            n_add = n_modify
            max_tries = 10 * n_add
            mod_arcs_set = set(mod_arcs)
            while added < n_add and tries < max_tries and possible_arcs:
                idx = random.randrange(len(possible_arcs))
                arc = possible_arcs[idx]
                # Try adding arc and check for cycles
                try:
                    test_arcs = list(mod_arcs_set) + [arc]
                    test_bn = pbn.KDENetwork(nodes=nodes, arcs=test_arcs)
                    test_bn.fit(sourcedat)
                    # If fit succeeds, accept the arc
                    mod_arcs_set.add(arc)
                    added += 1
                except Exception:
                    pass  # Adding this arc creates a cycle or is otherwise invalid
                possible_arcs.pop(idx)
                tries += 1
            final_arcs = list(mod_arcs_set)
            net_mod = pbn.KDENetwork(nodes=nodes, arcs=final_arcs)
            net_mod.fit(sourcedat)
            
            # 4. Compare structures
            added_arcs = [arc for arc in final_arcs if arc not in orig_arcs]
            removed_arcs = [arc for arc in orig_arcs if arc not in final_arcs]
            nodemap = {node: i for i, node in enumerate(net_mod.nodes())}
            hmd = hamming_distance(net_mod.arcs(), net_orig.arcs(), nodemap)
            
            print("Number of arcs in original:", len(orig_arcs))
            print("Number of arcs in modified:", len(final_arcs))
            print("Number of arcs added:", len(added_arcs))
            print("Number of arcs removed:", len(removed_arcs))
            print("Added arcs:", added_arcs)
            print("Removed arcs:", removed_arcs)
            print("Hamming distance:", hmd)
            
            # 5. save arcs sample from modified network
            new_samples = net_mod.sample(sample_size).to_pandas()
            new_arcs = pd.DataFrame(final_arcs, columns=['from', 'to'])
            new_samples.to_csv(spath + f"/{prefix}_{basename}_{int(percent_to_modify*100)}p_sampled.csv", index=False)
            new_arcs.to_csv(spath + f"/{prefix}_{basename}_{int(percent_to_modify*100)}p_arcs.csv", index=False)
    

def train_reference(path, trainsample, testsample, sspath, alpha=0.05, kcv=5, seed=15, **kwargs):
    data = pd.read_csv(path).sample(trainsample+testsample, random_state=seed)
    nodes = data.columns.values
    traindata = data[:trainsample]
    traindata.to_csv(f"{sspath}_train.csv", index=False)
    
    testdata = data[trainsample:]
    testdata.to_csv(f"{sspath}_test.csv", index=False)
    
    #Train HC model
    print(f"\n### Training HC model on {sspath} ###")
    pool = pbn.OperatorPool([pbn.ArcOperatorSet()])
    start_model = pbn.KDENetwork(nodes=nodes)
    score = pbn.ValidatedLikelihood(df=traindata, k=kcv)
    modelSCORE = pbn.GreedyHillClimbing().estimate(pool, score, start_model, **kwargs)
    modelSCORE.fit(traindata)    
    draw_model(modelSCORE, f"{sspath}_HC")

    ## Train PC model
    print(f"\n### Training PC model on {sspath} ###")
    rcot = pbn.RCoT(traindata)
    pdag = pbn.PC().estimate(rcot, alpha=alpha)
    try:
        dag = pdag.to_dag()
    except ValueError:
        dag = pdag.to_approximate_dag()
    modelCONSTRAINT = pbn.KDENetwork(dag)
    modelCONSTRAINT.fit(traindata)
    draw_model(modelCONSTRAINT, f"{sspath}_PC")
    
    return modelSCORE, modelCONSTRAINT, traindata

# Example usage:
listdirs = sorted(glob.glob("public_datasets/uci_ready/*.csv"))
for path in list(listdirs):
    modify_arcs_and_sample(path, trainsample=10000, testsample=1024, sample_size=40000, percentages=[0.05, 0.1, 0.2], 
                        output_dir='uci_nets', seed=15, alpha=0.05, kcv=5, patience=3, max_indegree=5)
