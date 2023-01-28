# Learning to Prune Instances of Steiner Tree Problem in Graphs
By Jiwei Zhang, Deepak Ajwani  
University College Dublin  
Datasets: http://steinlib.zib.de/steinlib.php  
Formulation: A dual ascent approach for steiner tree problems on a directed graph (1984)  

Folder Structure:
- df 
    - iXXX-AAA.csv
- ds
    - iXXX
        - iXXX-AAA.stp
- log
    - iXXX
        - iXXX-AAA_log.txt
    - process
        - logs that we get when we run our code. (1 for original problem solving and 1 for pipeline)
- src
    - script
        - formulation.py
        - functions.py
        - pruning.py
    - main.py
- README.md

For df folder:
    This folder is to save all features of each problem instance used in the train/test of the project
    XXX is the problem set name, either 080 or 160.
    AAA is the problem ID in the problem set.

For ds folder:
    This folder is to save all original problem instances downloaded from SteinLib website. 
    XXX is the problem set name, either 080 or 160.
    AAA is the problem ID in the problem set.

For log folder:
    log folder is to save the log which we generated to record the ILP and LP solutions.
    The generating function is saved in function.py
    the format of log is:
        - Runtime to get ILP solution
        - ILP objective value
        - Runtime to get LP solution
        - LP objective value
        - Whether LP objective is the same with ILP objective
        - Break line
        - Edges in ILP solution
        - ...
        - Break line
        - Edges in LP solution with values
        - ...

For src folder:
    The script folder is to save all the scripts we used.
        1: function.py has all the functions we used. (util and helper)
        2: formulation.py has the formulation we used.

main.py is the pipeline script to re-run the code and get our result.

Dependencies:
        pip install gurobipy, networkx
        You need a valid Gurobi license to run this code


