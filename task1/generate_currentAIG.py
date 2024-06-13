# import pickle

# with open('project_data/adder_1000.pkl', 'rb') as f:
#     data = pickle.load(f)

# print(data)
import os
import pickle
from tqdm import tqdm
import random
for filename in tqdm(os.listdir('project_data'), desc='Processing files', unit='files'):
    if filename.endswith('.pkl'):
        if random.random() > 0.01:
            continue
        with open('project_data/' + filename, 'rb') as f:
            data = pickle.load(f)
            for i in range(len(data['input'])):            
                state = data['input'][i]
                circuitName, actions = state.split('_')
                circuitPath = '/mnt/d/交大学习/机器学习/大作业/project/InitialAIG/train/' + circuitName + '.aig'
                libFile = '/mnt/d/交大学习/机器学习/大作业/project/lib/7nm/7nm.lib'
                logFile = circuitName + '.log'
                nextState = state + '.aig'  # current AIG file
                nextStatelabel = data['target'][i]
                nextStatePath = os.path.join('/mnt/d/交大学习/机器学习/大作业/project/currentAIG/aig', nextState)
                if os.path.exists(nextStatePath):
                    continue 
                nextStatelabelPath = os.path.join('/mnt/d/交大学习/机器学习/大作业/project/currentAIG/label', state + '.pkl')
                with open(nextStatelabelPath, 'wb') as f:
                    pickle.dump(nextStatelabel, f)
                synthesisOpToPosDic = {
                    0: 'refactor',
                    1: 'refactor -z',
                    2: 'rewrite',
                    3: 'rewrite -z',
                    4: 'resub',
                    5: 'resub -z',
                    6: 'balance'
                }

                action_cmd = ''
                for action in actions:
                    action_cmd += synthesisOpToPosDic[int(action)] + ';'
                abcRunCmd = "/usr/local/bin/yosys-abc -c \" read "+ circuitPath + ";" + action_cmd + "; read_lib " + libFile + "; write " + nextStatePath + "; print_stats\" > " + logFile
                os.system(abcRunCmd)
