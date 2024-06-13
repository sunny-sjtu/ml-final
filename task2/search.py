# coding=utf-8
import os
import torch
import shutil
import abc_py
import numpy as np
import pickle
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch_geometric.nn


synthesisOpToPosDic = {
    0: 'refactor',
    1: 'refactor -z',
    2: 'rewrite',
    3: 'rewrite -z',
    4: 'resub',
    5: 'resub -z',
    6: 'balance'
}

libFile = '7nm.lib'
Test_dir = 'test'
device = "cpu"


class GATNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=True, dropout=0.6)
        self.pool = torch_geometric.nn.global_mean_pool

    def forward(self, x, edge_index, batch):
        # print(x.shape, edge_index.shape)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.pool(x, batch)
        return x.squeeze(1)
        
model1 = GATNet(input_dim=1, hidden_dim=32, output_dim=1).to(device)
model1.load_state_dict(torch.load("1.pth",map_location=torch.device('cpu')))
model2 = GATNet(input_dim=1, hidden_dim=32, output_dim=1).to(device)
model2.load_state_dict(torch.load("2.pth",map_location=torch.device('cpu')))


def move_to_device(data, device):

    if isinstance(data, torch.Tensor):
        return data.to(device)

    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}

    elif isinstance(data, (list, tuple)):
        return [move_to_device(x, device) for x in data]
    return data

# input: list 
# output: max value's index in list
def argmax(childScores):
    max_value = max(childScores)
    return childScores.index(max_value)

# AIG TO DATA 
# input: AIG 
# output: data(map)   
def represent(state):
    #print(state)
    _abc = abc_py.AbcInterface()
    _abc.start()
    _abc.read(state)
    data = {}
    numNodes = _abc.numNodes()
    data['node_type'] = np.zeros(numNodes, dtype = int)
    data['num_inverted_predecessors'] = np.zeros(numNodes, dtype = int)
    edge_src_index = []
    edge_target_index = []
    for nodeIdx in range(numNodes):
        aigNode = _abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        data['num_inverted_predecessors'][nodeIdx] = 0
        if nodeType == 0 or nodeType == 2:
            data['node_type'][nodeIdx] = 0
        elif nodeType == 1:
            data['node_type'][nodeIdx] = 1
        else:
            data['node_type'][nodeIdx] = 2
            if nodeType == 4:
                data['num_inverted_predecessors'][nodeIdx] = 1
            if nodeType == 5:
                data['num_inverted_predecessors'][nodeIdx] = 2
        if (aigNode.hasFanin0()):
            fanin = aigNode.fanin0()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
        if (aigNode.hasFanin1()):
            fanin = aigNode.fanin1()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
    data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype = torch.long)
    data['node_type'] = torch.tensor(data['node_type'])
    data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
    data['nodes'] = numNodes
    data['label'] = 0
    return data

# input: aiglist(7 aig) 
# output: scorelist(7 numpy) 
def Evaluation(AIG_list, n = 0.5):
    if n < 0 or n > 1:
        print("?")
        return 404
    model1.eval()
    model2.eval()
    #print('model_count')
    predictions = []
    with torch.no_grad():
        for AIG in AIG_list:
            #print(AIG)
            #state = AIG.split('.')[0]
            #print(state)
            data = represent(AIG)
            with open(AIG.replace('.aig', '.pkl'), 'wb') as f:
                pickle.dump(data, f)
            with open(AIG.replace('.aig', '.pkl'), 'rb') as f:
                data = pickle.load(f)
            test_dataset = []
            test_dataset.append(Data(x=data['node_type'].unsqueeze(1).float(), edge_index=data['edge_index'], y=data['label']))
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            with torch.no_grad():
                for data in test_loader:
                    data = move_to_device(data, device)
                    output1 = model1(data.x, data.edge_index, data.batch)
                    output2 = model1(data.x, data.edge_index, data.batch)
                    output = output1*n + output2*(1-n)
                    #print(output1,output2,output)
                    predictions.extend(output.cpu().numpy())
    return predictions

# input: test_dir(location) 
# output: actionlist(str)
def greedy(filename):
    if filename.endswith('.aig'):
        circuit_name = filename.split('.')[0]
        logFile = filename.replace('.aig', '.log')
        actionlist = ""
        childs = []
        for child in range(7):
            childFile = circuit_name + '_' + str(child) + '.aig'
            #print(childFile)
            fpath = "./test/"+filename
            abcRunCmd = "/usr/local/bin/yosys-abc -c \" read "+ fpath + ";" + synthesisOpToPosDic[child] + "; read_lib " + libFile + "; write " + childFile + "; print_stats\" > " + logFile
            #print(abcRunCmd)
            os.system(abcRunCmd)
            childs.append(childFile)
        #print(childs) 
        childScores = Evaluation(childs)
        #print(childScores)
        action = argmax(childScores)
        filename = childs[action]
        actionlist = actionlist + str(action)
        
        for step in range(9):
            childs = []
            for child in range(7):
                childFile = filename.split('.')[0] + str(child) + '.aig'
                #print(childFile)
                fpath = filename
                abcRunCmd = "/usr/local/bin/yosys-abc -c \" read "+ fpath + ";" + synthesisOpToPosDic[child] + "; read_lib " + libFile + "; write " + childFile + "; print_stats\" > " + logFile
                #print(abcRunCmd)
                os.system(abcRunCmd)
                childs.append(childFile)
            #print(childs) 
            childScores = Evaluation(childs)
            #print(childScores)
            action = argmax(childScores)
            filename = childs[action]
            actionlist = actionlist + str(action)
        return actionlist
    
def n_step_greedy(filename, n):
    if filename.endswith('.aig'):
        circuit_name = filename.split('.')[0]
        logFile = filename.replace('.aig', '.log')
        actionlist = ""
        
        for step in range(0, 10, n):
            childs = [filename]
            # Generate 7^n different AIG files
            for _ in range(n):
                new_childs = []
                for parent in childs:
                    for child in range(7):
                        childFile = parent.split('.')[0] + '_' + str(child) + '.aig'
                        fpath = "./test/" + parent if step == 0 else parent
                        abcRunCmd = f"/usr/local/bin/yosys-abc -c \" read {fpath}; {synthesisOpToPosDic[child]}; read_lib {libFile}; write {childFile}; print_stats\" > {logFile}"
                        os.system(abcRunCmd)
                        new_childs.append(childFile)
                childs = new_childs
            
            # Evaluate all generated AIG files
            childScores = Evaluation(childs)
            action_index = argmax(childScores)
            filename = childs[action_index]
            
            # Determine the action sequence that led to the best scoring AIG
            action_sequence = ""
            for i in range(n):
                action = (action_index // (7 ** (n - i - 1))) % 7
                action_sequence += str(action)
            
            actionlist += action_sequence
        
        return actionlist   
    
def simulated_annealing(filename, initial_temperature, cooling_rate):
    def acceptance_probability(old_score, new_score, temperature):
        if new_score > old_score:
            return 1.0
        else:
            return math.exp((new_score - old_score) / temperature)

    if filename.endswith('.aig'):
        circuit_name = filename.split('.')[0]
        logFile = filename.replace('.aig', '.log')
        actionlist = ""
        
        # Initialize the current state
        current_filename = filename
        current_score = Evaluation([current_filename])[0]
        
        temperature = initial_temperature

        for step in range(10):
            if temperature <= 0:
                break

            childs = []
            for child in range(7):
                childFile = current_filename.split('.')[0] + '_' + str(child) + '.aig'
                fpath = "./test/" + current_filename if step == 0 else current_filename
                abcRunCmd = f"/usr/local/bin/yosys-abc -c \" read {fpath}; {synthesisOpToPosDic[child]}; read_lib {libFile}; write {childFile}; print_stats\" > {logFile}"
                os.system(abcRunCmd)
                childs.append(childFile)
            
            childScores = Evaluation(childs)
            best_action = argmax(childScores)
            best_child = childs[best_action]
            best_score = childScores[best_action]
            
            random_action = random.randint(0, 6)
            random_child = childs[random_action]
            random_score = childScores[random_action]

            if acceptance_probability(current_score, random_score, temperature) > random.random():
                current_filename = random_child
                current_score = random_score
                actionlist += str(random_action)
            else:
                current_filename = best_child
                current_score = best_score
                actionlist += str(best_action)

            temperature *= cooling_rate

        return actionlist       
for file in os.listdir(Test_dir):
    path = greedy(file)
    print(file,path)