Combining pipelines that I found either too simple or too complicated for task/data training of different RNNs and possible other models on neural tasks. 

Ultimate goal: train gnODEs on real neural datasets and extract both connectivity and dynamics. Also push the limits of efficient network training. 

Current state: defining efficient data loading to the task training, getting flow fields, fitting on *trajectories* (not sim data yet)

When launching Ray sessions from the command line run:
python -m main.train[_cluster].py

Jupyter notebooks have added the sys.path explicitly
