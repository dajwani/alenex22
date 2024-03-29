######################################################################################
# Import
######################################################################################
import pandas as pd
import numpy as np
import re
import os
import networkx as nx
import timeit
from script.formulation import *
from script.pruning import *
from sklearn.model_selection import train_test_split
######################################################################################
# 
######################################################################################
def read_df(file_name):
    df = pd.read_csv("./df/"+file_name+".csv")
    return df

def save_df(df, file_name):
    df.to_csv("./df/"+file_name+".csv", index=False)

def read_graph_directed(filename):
    with open(filename) as f:
        lines = f.readlines()
        arcs = []
        for line in lines:
            if line == '\n': 
                continue
            parts = line.split()
            det = parts[0]
            if det == 'Name':
                name = parts[1]
            elif det == 'Nodes':
                n_vertices = int(parts[1])
            elif det == 'Edges':
                n_edges = int(parts[1])
            elif det == 'E':
                i = int(parts[1])
                j = int(parts[2])
                c = int(parts[3])
                arcij = ((i,j),c)
                arcji = ((j,i),c)
                arcs.append(arcij)
                arcs.append(arcji)
            elif det == 'Terminals':
                n_terminals = int(parts[1])
        vertices = np.arange(1, int(n_vertices)+1)
        vertices = vertices.tolist()
        terminals = np.arange(1, int(n_terminals)+1)
        terminals = terminals.tolist()
        assert(int(n_edges) == len(arcs)/2)
    f.close()
    return [vertices, arcs, terminals]

def read_graph_undirected(filename):
    with open(filename) as f:
        lines = f.readlines()
        edges = []
        for line in lines:
            if line == '\n': 
                continue
            parts = line.split()
            det = parts[0]
            if det == 'Name':
                name = parts[1]
            elif det == 'Nodes':
                n_vertices = int(parts[1])
            elif det == 'Edges':
                n_edges = int(parts[1])
            elif det == 'E':
                i = int(parts[1])
                j = int(parts[2])
                c = int(parts[3])
                edge = ((i,j),c)
                edges.append(edge)
            elif det == 'Terminals':
                n_terminals = int(parts[1])
        vertices = np.arange(1, int(n_vertices)+1)
        vertices = vertices.tolist()
        terminals = np.arange(1, int(n_terminals)+1)
        terminals = terminals.tolist()
        assert(int(n_edges) == len(edges))
    f.close()
    return [vertices, edges, terminals]
######################################################################################
#
######################################################################################
def generate_log(filename):
    # v: vertices; c: cost; r: runtime.
    graph = read_graph_directed(filename)
    ilp_v, ilp_c, ilp_r = ILP(graph)
    lp_v, lp_c, lp_r = LP(graph)
    f = open("./log/"+filename+"_log.txt", "wt")
    f.write(str(ilp_r) + "\n")
    f.write(str(ilp_c) + "\n")
    f.write(str(lp_r) + "\n")
    f.write(str(lp_c) + "\n")
    f.write(str(lp_c == ilp_c))
    f.write("\n")   
    for each in ilp_v:
        f.write(str(each)+ "\n")
    f.write("\n")
    for each in lp_v:
        f.write(str(each)+ "\n")

def read_log(filename):
    with open(filename) as f:
        lines = [line.strip() for line in f.readlines()]
        ilp_rt = lines[0]
        ilp_c = lines[1]
        lp_rt = lines[2]
        lp_c = lines[3]
        # lines[4] is whether ilp_c == lp_c
        sols = lines[5:]
        b = sols.index('')
        ilp_sol = [re.sub("[()',]"," ", term).split() for term in sols[:b]]
        lp_sol = [re.sub("[()',]"," ", term).split() for term in sols[b+1:]]
    return {"ilp_rt" : ilp_rt, "ilp_c" : ilp_c, "ilp_sol" : ilp_sol, "lp_rt" : lp_rt, "lp_c" : lp_c, "lp_sol" : lp_sol}

######################################################################################
#
######################################################################################
# Check whether the given file has different LP and ILP result
def file_selection(filename):
    with open(filename) as f:
        if f.readlines()[4].startswith("F"): return True
        f.close()
    return False

# Get all selected files 
def get_selected_files():
    filenames = []
    log_filenames = os.listdir("../log/i080")
    filenames = []
    for filename in log_filenames:
        if file_selection("../log/i080/"+filename):
            filenames.append(filename.split('_')[0])
    log_filenames = os.listdir("../log/i160")
    for filename in log_filenames:
        if file_selection("../log/i160/"+filename):
            filenames.append(filename.split('_')[0])
    return filenames

######################################################################################
#
######################################################################################
# Get all paths for selected files
def get_paths(filenames, type=None):
    paths = []
    if type == 'df':
        for each in filenames:
            size = each.split('-')[0]
            path = "../df/"+size+"/"+each+'.csv'
            paths.append(path)
    elif type == 'ds':
        for each in filenames:
            size = each.split('-')[0]
            path = "../ds/"+size+"/"+each+'.stp'
            paths.append(path)
    elif type == 'log':
        for each in filenames:
            size = each.split('-')[0]
            path = "../log/"+size+"/"+each+'_log.txt'
            paths.append(path)
    else:
        print('ERROR: Undefined Type: ', type)
    return paths

######################################################################################
#
######################################################################################
def split_x_y(df):
    x = df.drop(columns=['Node 1','Node 2','Weight','ILP'])
    y = df['ILP']
    return x, y

######################################################################################
# Feature functions 
#
# (All features are normalized to 0 mean and 1 std)
#
# Feature 1: LP Value
# Feature 3: Weight (mean-std) 
#
# Feature 4: Local Rank min 
# Feature 5: Local Rank max
# Feature 6: Local Rank product
# Feature 7：Number of neighbors (Edges Connected)
#
# Feature 8: Degree centrality min
# Feature 9: Degree centrality max
# Feature 10: Degree centrality product
#
# Feature 11: Betweenness centrality min
# Feature 12: Betweenness centrality max
# Feature 13: Betweenness centrality product
#
# Feature 14: Eigenvector centrality min (Not Used)
# Feature 15: Eigenvector centrality max (Not Used)
# Feature 16: Eigenvector centrality product (Not Used)
######################################################################################
def normalize(series, isStd=False):
    if isStd:
    # Normalization by Standard Deviation
        miu = series.mean()
        std = series.std()
        result = (series-miu)/std
    else:
    # Default Normailzation
        min = series.min()
        max = series.max()
        delta = max-min
        result = (series-min)/delta
    return result

def localrank(node1, node2, df, minmax):
    # Find all edges connected to the target node
    target = node1
    node = node2
    tmp_df = df.loc[(df['Node 1'] == target) | (df['Node 2'] == target)]
    # Rank them by the weights
    tmp_df = tmp_df.sort_values(by=['Weight'],ignore_index=True, ascending=True)
    # Locate target edge's index
    index1 = tmp_df.index[
        (tmp_df['Node 1']==node) | (tmp_df['Node 2']==node)].tolist()[0]

    target = node2
    node = node1
    tmp_df = df.loc[(df['Node 1'] == target) | (df['Node 2'] == target)]
    tmp_df = tmp_df.sort_values(by=['Weight'],ignore_index=True, ascending=True)
    index2 = tmp_df.index[
        (tmp_df['Node 1']==node) | (tmp_df['Node 2']==node)].tolist()[0]
    
    if minmax == 'min':
        return min(index1, index2)
    if minmax == 'max':
        return max(index1, index2)
    print("ERROR: Undefined type for minmax: ", minmax)
    return None

def  get_connected(df, node1, node2):
    count = 0
    count += len(df.loc[(df['Node 1'] == node1)|(df['Node 2'] == node1)])
    count += len(df.loc[(df['Node 1'] == node2)|(df['Node 2'] == node2)])
    count -= 2
    return count

def dataframe_generate(ds_filename, log_filename):
    vertices, edges, terminals = read_graph_undirected(ds_filename)
    series = []
    df = pd.DataFrame(columns = ['Node 1', 'Node 2', 'Weight'])
    for edge in edges:
        nodes = edge[0]
        series.append({'Node 1' : nodes[0] , 'Node 2' : nodes[1], 'Weight' : edge[1]})
    df = pd.DataFrame(columns = ['Node 1', 'Node 2', 'Weight'], data=series)
    log = read_log(log_filename)

    # Feature: LP Value
    lp = log['lp_sol']
    ilp = log['ilp_sol']
    df['ILP'] = df.apply(
        lambda row : get_ILP(row['Node 1'], row['Node 2'], ilp),
        axis=1
    )

    df['LP'] = normalize(df.apply(
        lambda row : get_LP(row['Node 1'], row['Node 2'], lp),
        axis=1
    ), True)

    df['LP_bool'] = df.apply(
        lambda row : 1 if get_LP(row['Node 1'], row['Node 2'], lp) > 0 else 0,
        axis=1
    )

    start = timeit.default_timer()
    # Feature: Normailized Weight
    # df['Normalized Weight'] = normalize(df['Weight'])
    df['Normalized Weight Std'] = normalize(df['Weight'], True)

    # Feature: Local Rank
    df['Local Rank Max'] = normalize(df.apply(
        lambda row : localrank(
            row['Node 1'], row['Node 2'], df[['Node 1', 'Node 2', 'Weight']], 'min'),
        axis=1
    ), True)
    

    df['Local Rank Min'] = normalize(df.apply(
        lambda row : localrank(
            row['Node 1'], row['Node 2'], df[['Node 1', 'Node 2', 'Weight']],'max'),
        axis=1
    ), True)

    df['Local Rank Product'] = normalize(df.apply(
        lambda row : row['Local Rank Min']*row['Local Rank Max'],
        axis=1
    ), True)

    df['Edges Connected'] = normalize(df.apply(
        lambda row : get_connected(df, row['Node 1'], row['Node 2']),
        axis=1
    ), True)


    # Create Graph object
    G = nx.Graph()
    for index, row in df.iterrows():
        G.add_edge(row['Node 1'],row['Node 2'])

    # Feature: Degree Centrality
    d_cen = nx.degree_centrality(G)
    df['Degree Centrality Max'] = normalize(df.apply(
        lambda row : max(d_cen[row['Node 1']], d_cen[row['Node 2']]), axis=1
    ), True)

    df['Degree Centrality Min'] = normalize(df.apply(
        lambda row : min(d_cen[row['Node 1']], d_cen[row['Node 2']]), axis=1
    ), True)

    df['Degree Centrality Product'] = normalize(df.apply(
        lambda row : row['Degree Centrality Min']*row['Degree Centrality Max'],
        axis=1
    ), True)

    # Feature: Betweenness Centrality
    b_cen = nx.betweenness_centrality(G)
    df['Betweenness Centrality Max'] = normalize(df.apply(
        lambda row : max(b_cen[row['Node 1']], b_cen[row['Node 2']]), axis=1
    ), True)

    df['Betweenness Centrality Min'] = normalize(df.apply(
        lambda row : min(b_cen[row['Node 1']], b_cen[row['Node 2']]), axis=1
    ), True)

    df['Betweenness Centrality Product'] = normalize(df.apply(
        lambda row : row['Betweenness Centrality Min']*row['Betweenness Centrality Max'],
        axis=1
    ), True)

    # # Feature: Eigenvector Centrality
    # e_cen = nx.eigenvector_centrality(G)
    # df['Eigenvector Centrality Max'] = normalize(df.apply(
    #     lambda row : max(e_cen[row['Node 1']], e_cen[row['Node 2']]), axis=1
    # ), True)

    # df['Eigenvector Centrality Min'] = normalize(df.apply(
    #     lambda row : min(e_cen[row['Node 1']], e_cen[row['Node 2']]), axis=1
    # ), True)

    # df['Eigenvector Centrality Product'] = normalize(df.apply(
    #     lambda row : row['Eigenvector Centrality Min']*row['Eigenvector Centrality Max'],
    #     axis=1
    # ), True)

    stop = timeit.default_timer()

    # Runtime for feature engineering:
    runtime = stop - start + float(log["lp_rt"])

    return df, runtime

# Function to get dataframe of all train and test samples
def get_dfs(test_samples, filenames, ds_paths, log_paths):
    train_list = []
    test_list = []

    for file in filenames:
        tmp_df, runtime = dataframe_generate(ds_paths[file], log_paths[file])

        if file in test_samples:
            test_list.append(tmp_df)
        else:
            train_list.append(tmp_df)
    df_train = pd.concat(train_list)
    df_test = pd.concat(test_list)
    return df_train, df_test

# Split x and y
def get_xy(dataframe):
    X, y = split_x_y(dataframe)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return x_train, y_train, x_test, y_test
######################################################################################

######################################################################################
def get_LP(i,j,lp):
    for each in lp:
        if i == int(each[0]) and j == int(each[1]) : return float(each[2])
        if i == int(each[1]) and j == int(each[0]) : return float(each[2])
    return 0

def get_ILP(i,j,ilp):
    for each in ilp:
        if i == int(each[0]) and j == int(each[1]) : return float(each[2])
        if i == int(each[1]) and j == int(each[0]) : return float(each[2])
    return 0

######################################################################################
# Solving functions:
######################################################################################
def solve_LP(ds_path, log_path):
    graph = read_graph_undirected(ds_path)
    print("Graph readed.")
    print("Calculating features...")
    df, fe_rt = dataframe_generate(ds_path, log_path)
    print("Feature calculation runtime (LP not included): ", fe_rt)
    df_pruned_lp = prune_lp(df)
    print("Pruning finished.")
    graph_lp = reconstruct(df_pruned_lp, graph[2])
    print("Pruning Method: LP")   
    print("Solving...")
    sol, obj_lp, rt = ILP(graph_lp)
    print("Solved.")
    print("The OBJ is: ", obj_lp)
    print("The Runtime is: ", rt)
    return obj_lp


def solve_ILP(clf, ds_path, log_path, pr):
    graph = read_graph_undirected(ds_path)
    print("Graph readed.")
    print("Calculating features...")
    df, fe_rt = dataframe_generate(ds_path, log_path)
    print("Feature calculation runtime (LP not included): ", fe_rt)
    df_pruned_ml = prune_ml(clf, df, pr)
    print("Pruning finished.")
    graph_ml = reconstruct(df_pruned_ml, graph[2])
    print("Pruning Method: ML")
    print("Solving...")
    sol, obj_ilp, rt = ILP(graph_ml)
    print("Solved.")
    print("The OBJ is: ", obj_ilp)
    print("The Runtime is: ", rt)
    return obj_ilp

def prune_ml(clf, df, pr):
    x,y = split_x_y(df)
    y_pred_proba = clf.predict_proba(x.drop(['LP_bool'], axis=1))[:,1]
    threshold = np.sort(y_pred_proba)[int(pr*len(y_pred_proba)):][0]
    y_pred = (y_pred_proba >= threshold).astype('int')
    df['Predict'] = y_pred
    df_pruned = df.loc[(df['Predict'] > 0) | (df['LP_bool'] == 1)]
    return df_pruned[['Node 1', 'Node 2', 'Weight']]

def prune_lp(df):
    df_pruned = df.loc[df['LP_bool'] == 1]
    return df_pruned[['Node 1', 'Node 2', 'Weight']]

def reconstruct(df, terminals):
    vertices, arcs = graph_generate(df)
    graph = (vertices, arcs, terminals)
    return graph

######################################################################################
# Generate a graph (vertices and arcs)
######################################################################################
def graph_generate(df):
    vertices = []
    arcs = []
    for index, row in df.iterrows():
        i = row['Node 1']
        j = row['Node 2']
        c = row['Weight']
        if i not in vertices : vertices.append(i) 
        if j not in vertices : vertices.append(j)
        arcs.append(((i,j),c))
        arcs.append(((j,i),c))
    return vertices, arcs

######################################################################################