# config for client selection

PRE_SELECTION_METHOD = ['Random', 'Cluster1', 'Cluster2', 'NumDataSampling', 'NumDataSampling_rep',
                        'Random_d', 'Random_d_smp']

# POST_SELECTION: 'Pow-d','AFL','MaxEntropy','MaxEntropySampling','MaxEntropySampling_1_p','MinEntropy',
#                 'GradNorm','GradSim','GradCosSim','OCS','DivFL','LossCurr','MisClfCurr'

NEED_SETUP_METHOD = ['Cluster1', 'Cluster2', 'Pow-d', 'NumDataSampling', 'NumDataSampling_rep',
                     'Random_d_smp', 'GradSim', 'GradCosSim',
                     'Powd_baseline0', 'Powd_baseline1', 'Powd_baseline2']


NEED_INIT_METHOD = ['Cluster2', 'OCS', 'DivFL']


CANDIDATE_SELECTION_METHOD = ['Pow-d', 'Powd_baseline0', 'Powd_baseline1', 'Powd_baseline2']


NEED_LOCAL_MODELS_METHOD = ['GradNorm', 'GradSim', 'GradCosSim', 'OCS', 'DivFL']


LOSS_THRESHOLD = ['LossCurr']


CLIENT_UPDATE_METHOD = ['DoCL']