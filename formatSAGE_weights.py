from comet_ml import Experiment
from sys import settrace
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import gif
from typing import Dict, List, Union

from json.tool import main
from memory_profiler import profile
from dataProcessing import *

import pandas as pd
from torch.nn import ReLU
from neo4j import GraphDatabase

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics

import numpy as np
from statistics import mean
import plotly.express as px
from sklearn import preprocessing
import coloredlogs, logging
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
import torch_sparse
from torch_geometric.loader import HGTLoader, NeighborLoader,ImbalancedSampler
from torch_geometric.nn import Linear, SAGEConv, Sequential, to_hetero, MetaPath2Vec
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv,GATv2Conv, Linear, SuperGATConv,HANConv
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HGTConv, Linear, HeteroLinear
from torch_geometric.nn import GATConv, Linear, to_hetero
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss
from tqdm import tqdm
import time
import wandb
from IPython.display import Image
import os
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel
from random import randint
from sentence_transformers import SentenceTransformer
import statistics
from category_encoders import *
from torch_geometric.explain import Explainer, GNNExplainer
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import k_hop_subgraph
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Create an experiment with your api key
experiment = Experiment(
    api_key="yO67iXhjD5FRQH0uKNVOkpCuq",
    project_name="MasterThesis",
    workspace="aftab571",
)

coder_model = AutoModel.from_pretrained('GanjinZero/UMLSBert_ENG')
coder_tokenizer = AutoTokenizer.from_pretrained('GanjinZero/UMLSBert_ENG')

pd.options.mode.chained_assignment = None  # default='warn'

torch.cuda.empty_cache()

mylogs = logging.getLogger(__name__)

num_of_neg_samples= 2000
num_of_pos_samples= 2000

seed = 786
data = None
print(seed)



class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = GATConv((-1,-1), 2,edge_dim=1,add_self_loops=False, heads=1)  # TODO  64
        #self.conv2 = GATConv((-1,-1), 2,edge_dim=1,add_self_loops=False, heads=1)
        #self.conv2 = GATv2Conv((-1,-1), 2,edge_dim=1,add_self_loops=False, heads=1)
        #self.in1 = torch.nn.BatchNorm1d(64)
        # self.conv2 = GATv2Conv((-1,-1), 2,edge_dim=1,add_self_loops=False, heads=1)  # TODO
        # self.in2 = torch.nn.InstanceNorm1d(-1)
        # self.conv3 = GATConv((-1,-1), 2)
        # self.lin1 = Linear(-1, 2)


    def forward(self, x, edge_index, edge_attr):
        #x = F.dropout(x, p=0.6, training=True)
    
        x,alpha = self.conv1(x=x, edge_index= edge_index, edge_attr= edge_attr,return_attention_weights=True)
        return x,alpha
       
        #x = F.elu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        #x = self.conv2(x, edge_index, edge_attr)
      

class HAN(torch.nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, mdata, hidden_channels=128, heads=8):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=0.6, metadata=mdata)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.lin(out['Admission'])
        return F.softmax(out,dim=1)

class SAGE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels,aggr):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = SAGEConv((-1,-1), 64,aggr=aggr)  # TODO  64
        self.conv2 = SAGEConv((-1,-1), out_channels,aggr=aggr)
        self.lin1 = Linear(-1, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.lin1(x)
        return x



def create_train_val_test_mask(df):
    X = df.iloc[:,df.columns != 'label']
    Y = df['label']
    mask =[]
    X_train_complete, X_test, y_train_complete, y_test = train_test_split(X,df['label'].values.tolist(),random_state=seed, test_size=0.3,stratify=Y)
    test_mask = torch.zeros(df.shape[0], dtype=torch.bool)
    test_mask[X_test.index] = True
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    for train_index, val_index in skf.split(X_train_complete, y_train_complete):
        #print("Train_complete:", train_index, "TEST:", test_index)
        # X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        # y_train, y_val = Y[train_index], Y[val_index]

        X_train, X_val = X_train_complete.iloc[train_index], X_train_complete.iloc[val_index]
        y_train = [y_train_complete[i] for i in train_index] 
        y_val = [y_train_complete[j] for j in val_index]

        #X_train, X_test, y_train, y_test = train_test_split(X,df['label'].values.tolist(), test_size=0.2,random_state=1,stratify=Y)
        train_mask = torch.zeros(df.shape[0], dtype=torch.bool)
        val_mask = torch.zeros(df.shape[0], dtype=torch.bool)

        print("y_train: ",Counter(y_train).values()) 
        print("y_val: ",Counter(y_val).values()) 
    

        train_mask[X_train.index] = True
        val_mask[X_val.index] = True

        conf_df = pd.DataFrame()
        conf_df['admmision_id']= X_test['admmision_id']
        conf_df['actual']= y_test

        obj={
            'train_mask_set': train_mask,
            'val_mask_set' :  val_mask,
            'admission': conf_df,
            'admission_test_lst':X_test['admmision_id'].to_list(),
            'admission_train_lst':X_train['admmision_id'].to_list(),
            'admission_val_lst':X_val['admmision_id'].to_list()
        }
        mask.append(obj)
    #print(mask)

    return mask,test_mask
  
 
    # X_train_complete, X_test, y_train, y_test = train_test_split(X,df['label'].values.tolist(), test_size=0.1,random_state=seed,stratify=Y)
    # X_train, X_val, y_trainval, y_testval = train_test_split(X_train_complete,y_train, test_size=0.1,random_state=seed,stratify=y_train)
    
    # print("y_train: ",Counter(y_train).values()) 
    # print("y_test: ",Counter(y_test).values()) 
    # print("y_trainval: ",Counter(y_trainval).values()) 
    # print("y_testval: ",Counter(y_testval).values()) 


    # train_mask = torch.zeros(df.shape[0], dtype=torch.bool)
    # test_mask = torch.zeros(df.shape[0], dtype=torch.bool)
    # val_mask = torch.zeros(df.shape[0], dtype=torch.bool)

    # train_mask[X.index] = True
    # #train_mask[X_train.index] = True
    # test_mask[X_test.index] = True
    # val_mask[X_val.index] = True


    # conf_df = pd.DataFrame()
    # conf_df['admmision_id']= X_test['admmision_id']
    # conf_df['actual']= y_test



    # return train_mask,val_mask,test_mask,conf_df

# def train(model,optimizer,criterion,data):
#       model.train()
#       optimizer.zero_grad()  # Clear gradients.
#       out = model(data.x_dict, data.edge_index_dict)  # Perform a single forward pass.
#       mask = data['Admission'].train_mask
#       loss = criterion(out['Admission'][mask], data['Admission'].y[mask])  # Compute the loss solely based on the training nodes. ['Admission']
#       #print(out['Admission'][mask].shape)
#       #print(data['Admission'].y[mask].shape)
#       loss.backward()  # Derive gradients.
#       optimizer.step()  # Update parameters based on gradients.
#       return loss


def train(model,optimizer,criterion,data,train_loader,device):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out,w = model(data.x_dict, data.edge_index_dict,data.edge_attr_dict)  # Perform a single forward pass.
    mask = data['Admission'].train_mask
    loss = criterion(out['Admission'][mask], data['Admission'].y[mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, out, w


def test(model,optimizer,criterion,mask,data,device):
      model.eval()
      out,w = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
      pred = out['Admission'].argmax(dim=1)  # Use the class with highest probability.
      correct = pred[mask] == data['Admission'].y[mask]  # Check against ground-truth labels.
      acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
      return acc,pred


def gnn_model_summary(model):
    
    model_params_list = list(model.named_parameters())
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new)
    print("----------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0] 
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>20}  {:>25} {:>15}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("----------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)
    for parameter in model.parameters(): 
        print(parameter)


def folder(x,y):
    #timenow = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    folderpath = os.path.join(os.getcwd(), "mimicImages", str(x),str(y))
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        print('Created:', folderpath)
    return folderpath+"/"

def encodeFreq_Lbas(df, mask_list, test_mask):
    df.to_csv('labsBFreqEnc.csv')
    train_ids = mask_list['admission_train_lst']
    test_ids = mask_list['admission_test_lst']
    val_ids = mask_list['admission_val_lst']
    df_train = df[df['hadm_id'].isin(train_ids)]
    df_test = df[df['hadm_id'].isin(test_ids)]
    df_val = df[df['hadm_id'].isin(val_ids)]
    newf = pd.DataFrame(columns=df.columns)
    unq_labs_train = df_train.lab_id.unique()
    unq_labs_test = df_test.lab_id.unique()
    unq_labs_val = df_val.lab_id.unique()
    df_labs_grp_train = df_train.groupby(['lab_id'])
    df_labs_grp_test = df_test.groupby(['lab_id'])
    df_labs_grp_val = df_val.groupby(['lab_id'])
    for lab in unq_labs_train:
        try:
            plt_result_train = df_labs_grp_train.get_group((lab))
            plt_result_train['value'] = plt_result_train['value'].astype(float)
            scaler = MinMaxScaler()
            plt_result_train['value'] = scaler.fit_transform(plt_result_train[['value']].to_numpy())
            newf = newf.append(plt_result_train, ignore_index=True)
        except Exception as e:
            print("Exception at (is not a float) appending:",str(e),str(lab))
            fe = plt_result_train.groupby("value")['label'].mean()
            # #fe_ = fe/len(plt_result)
            plt_result_train['value'] = plt_result_train["value"].map(fe).round(4)
            plt_result_train['value'] = plt_result_train['value'].astype(float)
            newf = newf.append(plt_result_train, ignore_index=True)
            

    for lab in unq_labs_test:
        try:
            plt_result_test = df_labs_grp_test.get_group((lab))
            plt_result_test['value'] = plt_result_test['value'].astype(float)
            scaler = MinMaxScaler()
            plt_result_test['value'] = scaler.fit_transform(plt_result_test[['value']].to_numpy())
            newf = newf.append(plt_result_test, ignore_index=True)
        except Exception as e:
            print("Exception at (is not a float) appending:",str(e),str(lab))
            fe = plt_result_test.groupby("value")['label'].mean()
            # #fe_ = fe/len(plt_result)
            plt_result_test['value'] = plt_result_test["value"].map(fe).round(4)
            plt_result_test['value'] = plt_result_test['value'].astype(float)
            newf = newf.append(plt_result_test, ignore_index=True)

    for lab in unq_labs_val:
        try:
            plt_result_val = df_labs_grp_val.get_group((lab))
            plt_result_val['value'] = plt_result_val['value'].astype(float)
            scaler = MinMaxScaler()
            plt_result_val['value'] = scaler.fit_transform(plt_result_val[['value']].to_numpy())
            newf = newf.append(plt_result_val, ignore_index=True)
        except Exception as e:
            print("Exception at (is not a float) appending:",str(e),str(lab))
            fe = plt_result_val.groupby("value")['label'].mean()
            # #fe_ = fe/len(plt_result)
            plt_result_val['value'] = plt_result_val["value"].map(fe).round(4)
            plt_result_val['value'] = plt_result_val['value'].astype(float)
            newf = newf.append(plt_result_val, ignore_index=True)
    newf.to_csv('labsAFreqEnc.csv')
    return newf

def encodeFreq_drugs(df, mask_list, test_mask):
    df.to_csv('drugsBFreqEnc.csv')
    train_ids = mask_list['admission_train_lst']
    test_ids = mask_list['admission_test_lst']
    val_ids = mask_list['admission_val_lst']
    df_train = df[df['adm_id'].isin(train_ids)]
    df_test = df[df['adm_id'].isin(test_ids)]
    df_val = df[df['adm_id'].isin(val_ids)]
    newf = pd.DataFrame(columns=df.columns)
    unq_labs_train = df_train.drug_name.unique()
    unq_labs_test = df_test.drug_name.unique()
    unq_labs_val = df_val.drug_name.unique()
    df_labs_grp_train = df_train.groupby(['drug_name'])
    df_labs_grp_test = df_test.groupby(['drug_name'])
    df_labs_grp_val = df_val.groupby(['drug_name'])
    for lab in unq_labs_train:
        try:
            plt_result_train = df_labs_grp_train.get_group((lab))
            plt_result_train['dosage_val'] = plt_result_train['dosage_val'].astype(float)
            scaler = MinMaxScaler()
            plt_result_train['dosage_val'] = scaler.fit_transform(plt_result_train[['dosage_val']].to_numpy())
            newf = newf.append(plt_result_train, ignore_index=True)
        except Exception as e:
            print("Exception at (is not a float) appending:",str(e),str(lab))
            fe = plt_result_train.groupby("dosage_val")['label'].mean()
            # #fe_ = fe/len(plt_result)
            plt_result_train['dosage_val'] = plt_result_train["dosage_val"].map(fe).round(4)
            plt_result_train['dosage_val'] = plt_result_train['dosage_val'].astype(float)
            newf = newf.append(plt_result_train, ignore_index=True)


    for lab in unq_labs_test:
        try:
            plt_result_test = df_labs_grp_test.get_group((lab))
            plt_result_test['dosage_val'] = plt_result_test['dosage_val'].astype(float)
            scaler = MinMaxScaler()
            plt_result_test['dosage_val'] = scaler.fit_transform(plt_result_test[['dosage_val']].to_numpy())
            newf = newf.append(plt_result_test, ignore_index=True)
        except Exception as e:
            print("Exception at (is not a float) appending:",str(e),str(lab))
            fe = plt_result_test.groupby("dosage_val")['label'].mean()
            # #fe_ = fe/len(plt_result)
            plt_result_test['dosage_val'] = plt_result_test["dosage_val"].map(fe).round(4)
            plt_result_test['dosage_val'] = plt_result_test['dosage_val'].astype(float)
            newf = newf.append(plt_result_test, ignore_index=True)


    for lab in unq_labs_val:
        try:
            plt_result_val = df_labs_grp_val.get_group((lab))
            plt_result_val['dosage_val'] = plt_result_val['dosage_val'].astype(float)
            scaler = MinMaxScaler()
            plt_result_val['dosage_val'] = scaler.fit_transform(plt_result_val[['dosage_val']].to_numpy())
            newf = newf.append(plt_result_val, ignore_index=True)
        except Exception as e:
            print("Exception at (is not a float) appending:",str(e),str(lab))
            fe = plt_result_val.groupby("dosage_val")['label'].mean()
            # #fe_ = fe/len(plt_result)
            plt_result_val['dosage_val'] = plt_result_val["dosage_val"].map(fe).round(4)
            plt_result_val['dosage_val'] = plt_result_val['dosage_val'].astype(float)
            newf = newf.append(plt_result_val, ignore_index=True)
        
    newf.to_csv('drugsAFreqEnc.csv')
    return newf

def encodeFreq_vitals(df, mask_list, test_mask):
    df.to_csv('vitalsBFreqEnc.csv')
    train_ids = mask_list['admission_train_lst']
    test_ids = mask_list['admission_test_lst']
    val_ids = mask_list['admission_val_lst']
    df_train = df[df['adm_id'].isin(train_ids)]
    df_test = df[df['adm_id'].isin(test_ids)]
    df_val = df[df['adm_id'].isin(val_ids)]
    newf = pd.DataFrame(columns=df.columns)
    unq_labs_train = df_train.itemid.unique()
    unq_labs_test = df_test.itemid.unique()
    unq_labs_val = df_val.itemid.unique()
    df_labs_grp_train = df_train.groupby(['itemid'])
    df_labs_grp_test = df_test.groupby(['itemid'])
    df_labs_grp_val = df_val.groupby(['itemid'])
    for lab in unq_labs_train:
        try:
            plt_result_train = df_labs_grp_train.get_group((lab))
            plt_result_train['value'] = plt_result_train['value'].astype(float)
            scaler = MinMaxScaler()
            plt_result_train['value'] = scaler.fit_transform(plt_result_train[['value']].to_numpy())
            newf = newf.append(plt_result_train, ignore_index=True)
        except Exception as e:
            print("Exception at (is not a float) appending:",str(e),str(lab))
            fe = plt_result_train.groupby("value")['label'].mean()
            # #fe_ = fe/len(plt_result)
            plt_result_train['value'] = plt_result_train["value"].map(fe).round(4)
            plt_result_train['value'] = plt_result_train['value'].astype(float)
            newf = newf.append(plt_result_train, ignore_index=True)

    for lab in unq_labs_test:
        try:
            plt_result_test = df_labs_grp_test.get_group((lab))
            plt_result_test['value'] = plt_result_test['value'].astype(float)
            scaler = MinMaxScaler()
            plt_result_test['value'] = scaler.fit_transform(plt_result_test[['value']].to_numpy())
            newf = newf.append(plt_result_test, ignore_index=True)
        except Exception as e:
            print("Exception at (is not a float) appending:",str(e),str(lab))
            fe = plt_result_test.groupby("value")['label'].mean()
            # #fe_ = fe/len(plt_result)
            plt_result_test['value'] = plt_result_test["value"].map(fe).round(4)
            plt_result_test['value'] = plt_result_test['value'].astype(float)
            newf = newf.append(plt_result_test, ignore_index=True)

    for lab in unq_labs_val:
        try:
            plt_result_val = df_labs_grp_val.get_group((lab))
            plt_result_val['value'] = plt_result_val['value'].astype(float)
            scaler = MinMaxScaler()
            plt_result_val['value'] = scaler.fit_transform(plt_result_val[['value']].to_numpy())
            newf = newf.append(plt_result_val, ignore_index=True)
        except Exception as e:
            print("Exception at (is not a float) appending:",str(e),str(lab))
            fe = plt_result_val.groupby("value")['label'].mean()
            # #fe_ = fe/len(plt_result)
            plt_result_val['value'] = plt_result_val["value"].map(fe).round(4)
            plt_result_val['value'] = plt_result_val['value'].astype(float)
            newf = newf.append(plt_result_val, ignore_index=True)
    newf.to_csv('vitalsAFreqEnc.csv')
    return newf


#@profile
def main():
    model = None
    prev_mask = None
    global dataset

    heatmaps=[]
    st_time_nodes = time.time()

    df_admission,df_diagnosis,df_drugs,df_labs,df_vitals,df_diagnosis_features,df_demo,df_output,lst_weights, df_drugs_features,df_labs_features,df_vitals_features,df_output_features,df_demo_features = getPreprocessData('Graph',edge_merge='',grp_aggr='mean')  
    # edge_merge='Nodes' -- for having all the features on the Nodes
    # edge_merge='EMerge' -- for merge multiple edges into one.

    end_time = time.time()

    experiment.log_table("admissions.csv",df_admission.head(5))
    experiment.log_table("diagnosis.csv",df_diagnosis.head(5))
    experiment.log_table("drugs.csv",df_drugs.head(5))
    experiment.log_table("vitals.csv",df_vitals.head(5))
    experiment.log_table("diagnosis_features_bert.csv",df_diagnosis_features.head(5))
    experiment.log_table("demography.csv",df_demo.head(5))
    experiment.log_table("output.csv",df_output.head(5))
    experiment.log_table("labs.csv",df_labs.head(5))
    df_labs.to_csv('LabsCorrelation.csv')
    print("Time for Preprocessing data (Graph): ",end_time-st_time_nodes)

    mask_list,test_mask = create_train_val_test_mask(df_admission)
    
   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    for aggr in ['sum']:
        print(aggr)
        iter_count=0
        for i in mask_list:
            if df_labs.shape[0]>0:
                df_labs = encodeFreq_Lbas(df_labs, i ,test_mask)
                df_labs = df_labs.reset_index(drop=True)
                df_labs['index_col'] = df_labs.index
                df_labs_UML= getLabUMLS(df_labs)
                df_lab_expand = pd.merge(df_labs, df_labs_UML, left_on='lab_id', right_on='lab_id', how='left')
                df_labs_features = expandEmbeddings(df_lab_expand)
            if df_drugs.shape[0]>0:
                df_drugs = encodeFreq_drugs(df_drugs, i ,test_mask)
                df_drugs = df_drugs.reset_index(drop=True)
                df_drugs['index_col'] = df_drugs.index
                df_drugs_UML= getDrugUMLS(df_drugs)
                df_drug_expand = pd.merge(df_drugs, df_drugs_UML, left_on='drug_name', right_on='drug_name', how='left')
                df_drugs_features = expandEmbeddings(df_drug_expand)
            if df_vitals.shape[0]>0:
                df_vitals = encodeFreq_vitals(df_vitals, i ,test_mask)
                df_vitals = df_vitals.reset_index(drop=True)
                df_vitals['index_col'] = df_vitals.index
                df_vitals_UML= getVitalsUMLS(df_vitals)
                df_vitals_expand = pd.merge(df_vitals, df_vitals_UML, left_on='itemid', right_on='itemid', how='left')
                df_vitals_features = expandEmbeddings(df_vitals_expand)
            if df_output.shape[0]>0:
                df_output = encodeFreq_vitals(df_output, i ,test_mask)
                df_output = df_output.reset_index(drop=True)
                df_output['index_col'] = df_output.index
                df_output_UML= getVitalsUMLS(df_output)
                df_output_expand = pd.merge(df_output, df_output_UML, left_on='itemid', right_on='itemid', how='left')
                df_output_features = expandEmbeddings(df_output_expand)
            iter_count = iter_count +1
            data = HeteroData()
            if df_labs.shape[0]>0:
                df_labs = df_labs.fillna(0)
                df_labs['label']= df_labs['label'].astype(float)
            if df_drugs.shape[0]>0:
                #df_drugs['drug_name']= df_drugs['drug_name'].astype(float)
                df_drugs['dosage_val']= df_drugs['dosage_val'].astype(float)

            #df_admission[['gender','age', 'adm_typ','ethnicity','marital','religion']].values

            df_admission.to_csv('adm.csv')
            data['Admission'].x = torch.tensor(df_admission.loc[:, ~df_admission.columns.isin(["hadm_id","patients","label","marital","ethnicity","religion","hadm_id","gender","age","output","adm_typ","admmision_id","index_col"])].values.tolist(), dtype = torch.float).to(device)  #'ethnicity','marital','religion','gender','age'
            data['Admission'].y =  torch.tensor(df_admission['label'].values, dtype = torch.long).to(device)
            data['Admission'].train_mask = i['train_mask_set'].to(device)
            data['Admission'].val_mask = i['val_mask_set'].to(device)
            data['Admission'].test_mask = test_mask



            # #data['Labs'].x = torch.tensor(df_labs[['fluid','category']].values, dtype = torch.float).to(device) #df_labs.loc[:, ~df_labs.columns.isin(['hadm_id','lab_name'])].values.tolist()
            #data['Labs'].x = torch.tensor(df_labs_features.values,dtype = torch.float).to(device)
            # ["label","marital","ethnicity","religion","fluid","lab_id","category","adm_id","gender","hadm_id","lab_name","age","index_col","Embeddings"]
            # # for RQ-3 features on Node
            #data['Labs'].x = torch.tensor(df_labs.loc[:, ~df_labs.columns.isin(['age','hadm_id','marital','ethnicity','religion','gender','label','adm_id','adm_typ','lab_name','fluid','category','lab_id','index_col','Embeddings','value'])].values.tolist(), dtype = torch.float).to(device)

            # data['Admission', 'has_labs', 'Labs'].edge_index = torch.tensor(df_labs[['adm_id','index_col']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            # data['Admission', 'has_labs', 'Labs'].edge_attr  = torch.tensor(df_labs[['value']].values.tolist(), dtype=torch.float).contiguous().to(device)

            # data['Labs', 'rev_has_labs', 'Admission'].edge_index = torch.tensor(df_labs[['index_col','adm_id']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            # data['Labs', 'rev_has_labs', 'Admission'].edge_attr  = torch.tensor(df_labs[['value']].values.tolist(), dtype=torch.float).contiguous().to(device)

            # data['Vitals'].x = torch.tensor(df_vitals_features.values,dtype = torch.float).to(device)
            # # data['Vitals'].x = torch.tensor(df_vitals[['name']].values, dtype = torch.float).to(device)
            # data['Admission', 'has_vitals', 'Vitals'].edge_index = torch.tensor(df_vitals[['hadm_id','index_col']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            # data['Admission', 'has_vitals', 'Vitals'].edge_attr  = torch.tensor(df_vitals[['value']].values.tolist(), dtype=torch.float).contiguous().to(device)

            # data['Vitals', 'rev_has_vitals', 'Admission'].edge_index = torch.tensor(df_vitals[['index_col','hadm_id']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            # data['Vitals', 'rev_has_vitals', 'Admission'].edge_attr  = torch.tensor(df_vitals[['value']].values.tolist(), dtype=torch.float).contiguous().to(device)
            
            data['Output'].x = torch.tensor(df_output_features.values,dtype = torch.float).to(device)
            # data['Output'].x = torch.tensor(df_output[['name']].values, dtype = torch.float).to(device)
            data['Admission', 'has_ouput', 'Output'].edge_index = torch.tensor(df_output[['hadm_id','index_col']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            data['Admission', 'has_ouput', 'Output'].edge_attr  = torch.tensor(df_output[['value']].values.tolist(), dtype=torch.float).contiguous().to(device)

            data['Output', 'rev_has_ouput', 'Admission'].edge_index = torch.tensor(df_output[['index_col','hadm_id']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            data['Output', 'rev_has_ouput', 'Admission'].edge_attr  = torch.tensor(df_output[['value']].values.tolist(), dtype=torch.float).contiguous().to(device)

            # # # # #print(df_drugs[['dosage_val']].values)
            # # # #data['Drugs'].x = torch.tensor(df_drugs[['drug_name','dosage_unit']].values, dtype = torch.float).to(device)
            # data['Drugs'].x = torch.tensor(df_drugs_features.values,dtype = torch.float).to(device)
            # data['Admission', 'has_drugs', 'Drugs'].edge_index = torch.tensor(df_drugs[['hadm_id','index_col']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            # data['Admission', 'has_drugs', 'Drugs'].edge_attr  = torch.tensor(df_drugs[['dosage_val']].values.tolist(), dtype=torch.float).contiguous().to(device)

            # data['Drugs', 'rev_has_drugs', 'Admission'].edge_index = torch.tensor(df_drugs[['index_col','hadm_id']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            # data['Drugs', 'rev_has_drugs', 'Admission'].edge_attr  = torch.tensor(df_drugs[['dosage_val']].values.tolist(), dtype=torch.float).contiguous().to(device)

            # # # # #df_diagnosis.iloc[:,4:].drop('index_col',axis=1).values
            # data['Diagnosis'].x = torch.tensor(df_diagnosis_features.values,dtype = torch.float).to(device)
            # data['Admission', 'has_diagnosis', 'Diagnosis'].edge_index = torch.tensor(df_diagnosis[['hadm_id','index_col']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            # data['Diagnosis', 'rev_has_diagnosis', 'Admission'].edge_index = torch.tensor(df_diagnosis[['index_col','hadm_id']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            
            # data['Demography'].x = torch.tensor(df_demo[['atype']].values.tolist(),dtype = torch.float).to(device)
            # data['Demography'].x = torch.tensor(df_demo_features.loc[:, ~df_demo_features.columns.isin(["start","end","atype","index_col","hadm_id"])].values.tolist(),dtype = torch.float).to(device)
            # data['Admission', 'has_same_demo', 'Demography'].edge_index = torch.tensor(df_demo[['start','index_col']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            # data['Demography', 'rev_same_demo', 'Admission'].edge_index = torch.tensor(df_demo[['index_col','start']].values.tolist(), dtype=torch.long).t().contiguous().to(device)

            data.num_node_features = 3
            data.num_classes = len(df_admission['label'].unique())
            #data = T.ToUndirected()(data.to(device))
            #data = T.NormalizeFeatures()(data.to(device))
            #data = T.RandomNodeSplit()(data)
            dataset = data.to(device)

            data = dataset.to(device)
            # if not os.path.exists('MIMICDataObj.pt'):
            #     torch.save(data,'MIMICDataObj.pt')
            # train_loader = NeighborLoader(
            #     data,
            #     # Sample 15 neighbors for each node and each edge type for 2 iterations:
            #     num_neighbors=[4] * 2,
            #     # Use a batch size of 128 for sampling training nodes of type "paper":
            #     batch_size=8,
            #     input_nodes=('Admission', data['Admission'].train_mask),
            # )
            # batch = next(iter(train_loader)) 
            sampler = ImbalancedSampler(data['Admission'].y, input_nodes=data['Admission'].train_mask)  #, input_nodes=data['Admission'].train_mask
            # print(data.edge_types)
            loader = NeighborLoader(data, input_nodes=('Admission', data['Admission'].train_mask), batch_size=128,
                             num_neighbors={key: [30] * 2 for key in data.edge_types})  #sampler=sampler
            
            if data:
                print(data)
                wandb.init(project="test-project", entity="master-thesis-luffy07")
                #model = HAN(in_channels=-1, out_channels=2, mdata=data.metadata())
                # if model is not None:
                #   model = model
                # else:
                #   model = SAGE(hidden_channels=128,out_channels=2,aggr=aggr)
                model = GAT(hidden_channels=32,heads=2)
                model = model.to(device)
                print(model)
                model = to_hetero(model, data.metadata(), aggr=aggr).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
                criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(lst_weights, dtype=torch.float)).to(device)             #weight=torch.tensor([0.15, 0.85])
                    # criterion =  FocalLoss(mode="binary", alpha=0.25, gamma=2)
                    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
                for epoch in range(1, 501):
                    loss,out,w = train(model,optimizer,criterion,data,loader,device)
                    if iter_count == len(mask_list):
                        wandb.log({"Training loss": loss})
                        experiment.log_metric("Training loss",loss,step=epoch)
                    train_acc,pred_train = test(model,optimizer,criterion,data['Admission'].train_mask,data,device)

                    val_acc,pred_val = test(model,optimizer,criterion,data['Admission'].val_mask,data,device)
                    test_acc,pred_test = test(model,optimizer,criterion,data['Admission'].test_mask,data,device)
                    if iter_count == len(mask_list):
                        wandb.log({'Train_acc':train_acc,'Validation_acc':val_acc,'Test_acc':test_acc})
                        experiment.log_metric("Train Accuracy",train_acc,step=epoch)
                        experiment.log_metric("Validation Accuracy",val_acc,step=epoch)
                        experiment.log_metric("Test Accuracy",test_acc,step=epoch)
                    #if epoch%100==0:
                    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
                        #print(scheduler.get_last_lr())
                    #scheduler.step()
                #gnn_model_summary(model)

                # for capturing edge weights does not work with batch 

                df_links_0 = pd.DataFrame(w['Admission'][0][0].cpu().numpy(), columns=['src'])
                df_links_1 = pd.DataFrame(w['Admission'][0][1].cpu().numpy(), columns=['dest'])
                df_weights = pd.DataFrame(w['Admission'][1].cpu().detach().numpy(), columns=['weights'])
                df_edge_links = pd.concat([df_links_0,df_links_1,df_weights],axis=1)
                #print(df_edge_links.sort_values(by='src', ascending=False))
                # for labs
                if df_labs.shape[0]>1:
                    df3 = pd.merge(df_edge_links, df_labs[['lab_id','hadm_id','index_col','lab_name','value']], left_on='src', right_on='index_col', how='left')

                # for drugs
                if df_drugs.shape[0]>1:
                    df3 = pd.merge(df_edge_links, df_drugs[['drug_name','hadm_id','index_col','dosage_val']], left_on='src', right_on='index_col', how='left')

                # for vitals
                if df_vitals.shape[0]>1:
                    df3 = pd.merge(df_edge_links, df_vitals[['itemid','hadm_id','index_col','name']], left_on='src', right_on='index_col', how='left')

                # for diagnosis
                if df_diagnosis.shape[0]>1:
                    df3 = pd.merge(df_edge_links, df_diagnosis[['title','hadm_id','index_col']], left_on='src', right_on='index_col', how='left')

                # for vitals
                if df_output.shape[0]>1:
                    df3 = pd.merge(df_edge_links, df_output[['itemid','hadm_id','index_col','name']], left_on='src', right_on='index_col', how='left')


               
                
                mask_train = data['Admission'].train_mask
                cf_matrix = confusion_matrix(data['Admission'].cpu().y[mask_train], pred_train[mask_train].cpu())
                print("train cfm: ",cf_matrix)

                mask_val = data['Admission'].val_mask
                cf_matrix = confusion_matrix(data['Admission'].cpu().y[mask_val], pred_train[mask_val].cpu())
                print("Validation cfm: ",cf_matrix)


                mask_test = data['Admission'].test_mask
                cf_matrix = confusion_matrix(data['Admission'].cpu().y[mask_test], pred_test[mask_test].cpu())
                print("test cfm: ",cf_matrix)
                experiment.log_confusion_matrix(data['Admission'].cpu().y[mask_test], pred_test[mask_test].cpu(),title="GAT")
                sensitivity = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
                print('Sensitivity : ', sensitivity )
                experiment.log_metric("GAT Sensitivity", sensitivity)

                specificity = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
                print('Specificity : ', specificity)
                experiment.log_metric("GAT Specificity", specificity)

                # explainer = GNNExplainer(model, epochs=200, return_type='log_prob')
                # node_idx = 10
                # node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index,
                #                                                 edge_weight=edge_weight)
                # ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, y=data.y)
                # plt.show()

                ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt= '.3g')

                ax.set_title('GAT Confusion Matrix\n\n')
                ax.set_xlabel('\nPredicted Values')
                ax.set_ylabel('Actual Values ')

                ## Ticket labels - List must be in alphabetical order
                ax.xaxis.set_ticklabels(['Survived','Died'])
                ax.yaxis.set_ticklabels(['Survived','Died'])

                ## Display the visualization of the Confusion Matrix.
                experiment.log_figure(figure_name="GAT_Conf.png")
                plt.savefig('GAT_conf.png', dpi=400)
                #plt.show()
        
                plt.clf()

                fpr, tpr, _ = metrics.roc_curve(data['Admission'].cpu().y[mask_test], pred_test[mask_test].cpu())
                auc = round(metrics.roc_auc_score(data['Admission'].cpu().y[mask_test], pred_test[mask_test].cpu()), 4)
                plt.plot(fpr,tpr,label="GAT, AUC="+str(auc))
                plt.legend()
                print("AUC : ",auc)
                experiment.log_figure(figure_name="GAT_AUC.png")
                plt.savefig('GAT_AUC.png', dpi=400)
                plt.clf()

                if df_labs.shape[0]>1:
                    df_grp = df3.groupby(['lab_id']).agg({'weights': 'sum','lab_name':'first'})
                    
                    #df_grp.set_index('diff',inplace=True)
                    df_grp = df_grp.sort_values(['weights'],ascending=False)
                    df_grp = df_grp[df_grp['weights']>=0.05]
                    df_grp['Percentile Rank'] = df_grp['weights'].rank(pct=True)
                    df_grp.to_csv('weights_labs.csv')
                    fig = px.bar(df_grp, x='lab_name', y='weights')
                    path = folder('matrix','weightDistribution')
                    fig.write_image(path+'weightPlotLabs'+str(iter_count)+".png",width=1000, height=350, scale=2)

                if df_drugs.shape[0]>1:
                    df_grp = df3.groupby(['drug_name']).agg({'weights': 'sum','drug_name':'first'})
                    
                    #df_grp.set_index('diff',inplace=True)
                    df_grp = df_grp.sort_values(['weights'],ascending=False)
                    df_grp = df_grp[df_grp['weights']>=0.05]
                    df_grp['Percentile Rank'] = df_grp['weights'].rank(pct=True)
                    df_grp.to_csv('weights_drugs.csv')
                    fig = px.bar(df_grp, x='drug_name', y='weights')
                    path = folder('matrix','weightDistribution')
                    fig.write_image(path+'weightPlotDrugs'+str(iter_count)+".png",width=1000, height=350, scale=2)

                if df_diagnosis.shape[0]>1:
                    df_grp = df3.groupby(['title']).agg({'weights': 'sum','title':'first'})
                    
                    #df_grp.set_index('diff',inplace=True)
                    df_grp = df_grp.sort_values(['weights'],ascending=False)
                    df_grp = df_grp[df_grp['weights']>=0.05]
                    df_grp['Percentile Rank'] = df_grp['weights'].rank(pct=True)
                    df_grp.to_csv('weights_diagnosis.csv')
                    fig = px.bar(df_grp, x='title', y='weights')
                    path = folder('matrix','weightDistribution')
                    fig.write_image(path+'weightPlotDiagnosis'+str(iter_count)+".png",width=1000, height=350, scale=2)

                if df_vitals.shape[0]>1:
                    df_grp = df3.groupby(['itemid']).agg({'weights': 'sum','name':'first'})
                    
                    #df_grp.set_index('diff',inplace=True)
                    df_grp = df_grp.sort_values(['weights'],ascending=False)
                    df_grp = df_grp[df_grp['weights']>=0.05]
                    df_grp['Percentile Rank'] = df_grp['weights'].rank(pct=True)
                    df_grp.to_csv('weights_vitals.csv')
                    fig = px.bar(df_grp, x='name', y='weights')
                    path = folder('matrix','weightDistribution')
                    fig.write_image(path+'weightPlotVitals'+str(iter_count)+".png",width=1000, height=350, scale=2)


                if df_output.shape[0]>1:
                    df_grp = df3.groupby(['itemid']).agg({'weights': 'sum','name':'first'})
                    
                    #df_grp.set_index('diff',inplace=True)
                    df_grp = df_grp.sort_values(['weights'],ascending=False)
                    df_grp = df_grp[df_grp['weights']>=0.05]
                    df_grp['Percentile Rank'] = df_grp['weights'].rank(pct=True)
                    df_grp.to_csv('weights_output.csv')
                    fig = px.bar(df_grp, x='name', y='weights')
                    path = folder('matrix','weightDistribution')
                    fig.write_image(path+'weightPlotOutput'+str(iter_count)+".png",width=1000, height=350, scale=2)

                # explanation_type=explanation_type,

                if False:
                    for explanation_type in ['phenomenon', 'model']:
                        explainer = Explainer(
                            model=model,
                            algorithm=GNNExplainer(epochs=300),
                            explainer_config = dict(
                                explanation_type = explanation_type,
                                node_mask_type='attributes',
                                edge_mask_type='object',
                            ),
                            model_config=dict(
                                mode='classification',
                                task_level='node',
                                return_type='raw',
                            ),
                        )

                        # Explanation ROC AUC over all test nodes:
                        targets, preds = [], []
                        node_indices = range(0, data['Admission'].cpu().num_nodes, 1)
                        for node_index in tqdm(node_indices, leave=False, desc='Train Explainer'):
                            target = data['Admission'].y if explanation_type == 'phenomenon' else None
                            explanation = explainer(data['Admission'].cpu().x, data['Admission', 'has_drugs', 'Drugs'].cpu().edge_index, index=node_index, target=target)

                            _, _, _, hard_edge_mask = k_hop_subgraph(node_index, num_hops=1,
                                                                    edge_index=data['Admission', 'has_drugs', 'Drugs'].cpu().edge_index)

                            targets.append(data['Admission'].edge_mask[hard_edge_mask].cpu())
                            preds.append(explanation.edge_mask[hard_edge_mask].cpu())

                        auc = roc_auc_score(torch.cat(targets), torch.cat(preds))
                        print(f'Mean ROC AUC (explanation type {explanation_type:10}): {auc:.4f}')

def seed_everything(seed=seed):                                                  
    #random.seed(seed)                                                            
    torch.manual_seed(seed)                                                      
    torch.cuda.manual_seed_all(seed)                                             
    np.random.seed(seed)                                                         
    os.environ['PYTHONHASHSEED'] = str(seed)                                     
    torch.backends.cudnn.deterministic = True                                    
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    seed_everything()
    main()