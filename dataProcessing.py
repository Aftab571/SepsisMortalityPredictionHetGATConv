from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Union
import pandas as pd
from neo4j import GraphDatabase
import numpy as np
from statistics import mean
from sklearn import preprocessing
import coloredlogs, logging
from tqdm import tqdm
import time
import wandb
import os
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel
from random import randint
from sentence_transformers import SentenceTransformer
import statistics
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from gensim.models.word2vec import Word2Vec
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA





coder_model = AutoModel.from_pretrained('GanjinZero/UMLSBert_ENG')
coder_tokenizer = AutoTokenizer.from_pretrained('GanjinZero/UMLSBert_ENG')

num_of_neg_samples= 50
num_of_pos_samples= 50



class Connection:
    
    def fetch_data(self,query, params={}):
        with self.driver.session() as session:
            result = session.run(query, params)
            #return result
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

    def __init__(self):
        self.driver = GraphDatabase.driver("bolt://127.0.0.1:17687", auth=("neo4j", "123456"))


def expandEmbeddings(df):
    df2 = pd.DataFrame(df['Embeddings'].tolist())
    pca = PCA(n_components=100)
    pca.fit(df2)
    df_pca = pd.DataFrame(pca.transform(df2), columns=['PCA%i' % i for i in range(100)], index=df.index)
    return df_pca

def split(word):
    return [char for char in word]

def Average(l): 
    avg = sum(l) / len(l) 
    return avg
     
# Driver code
def evaluted_val(word):
    num=None
    try:
        word = word.upper()
        if word.startswith("O1B"):
            word = word.replace("O1B",'')
            num= float(word)
        if word.startswith("/"):
            iword = word.split("/")
            iword = ' '.join(iword).split()
            iword.append(0.0)
            iword = [float(i) for i in iword]
            #print("startswith",iword)
            num= statistics.harmonic_mean(iword)
        if word.endswith("/"):
            iword = word.split("/")
            iword = ' '.join(iword).split()
            iword.append(0.0)
            iword = [float(i) for i in iword]
            #print("endswith",iword)
            num= statistics.harmonic_mean(iword)
        if word.startswith("GREATER THAN"):
            word = word.replace("GREATER THAN ",'>')
        if word.startswith("LESSTHAN"):
            word = word.replace("LESSTHAN ",'<')
        if word.startswith("LESS THJAN"):
            word = word.replace("LESS THJAN ",'<')
        if word.startswith("LESS THEN"):
            word = word.replace("LESS THEN ",'<')
        if word.startswith("LESS THASN"):
            word = word.replace("LESS THASN ",'<')
        if word.startswith("LESS THAM"):
            word = word.replace("LESS THAM ",'<')
        if word.startswith("LESS TAHN"):
            word = word.replace("LESS TAHN ",'<')
        if word.startswith("LES THAN"):
            word = word.replace("LES THAN ",'<')
        if word.startswith("GREATER THASN"):
            word = word.replace("GREATER THASN ",'>')
        if word.startswith("GREATER THEN"):
            word = word.replace("GREATER THEN ",'>')
        if word.startswith("LESS THAN"):
            word = word.replace("LESS THAN ",'<')
        if word.endswith("ONE"):
            word = word.replace("ONE",'1')
        if word.endswith("FIVE"):
            word = word.replace("FIVE",'5')
        if word.startswith(">GREATER THAN"):
            word = word.replace(">GREATER THAN ",'>')
        if word.endswith("NG/ML"):
            word = word.replace("NG/ML",'')
        if word.endswith(" C"):
            word = word.replace(" C",'')
            
        
        arr= split(word)
        #print(arr)
        if arr[0] == '>':
            num= ''.join(arr[1:])
            num = num.replace(',','')
            num = num.replace('=','')
            #print(num)
            num= float(num)
            if num<=1:
                num=num+0.1
            else:
                num=num+1
        if arr[0] == '<':
            num= ''.join(arr[1:])
            num = num.replace(',','')
            num = num.replace('=','')
            num= float(num)
            if num<=1:
                num=num-0.1
            else:
                num=num-1
        if arr[0]== "=":
            num= ''.join(arr[1:])
            num = num.replace(',','')
            num= float(num)
        if word.find('/') != -1:
            arr = word.split('/')
            arr = [float(i) for i in arr]
            arr.sort(reverse=True)
            num = statistics.harmonic_mean(arr)

        if word.find('-') != -1:
            arr = word.split('-')
            arr = [float(i) for i in arr]
            num = Average(arr)

        if word.find(',') != -1:
            word = word.replace(',','')
            num= float(word)
        
        
    except Exception as e:
        print(str(e))
        #continue 
    return num

def isfloat(x):
    try:
        a = float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True

def isint(x):
    try:
        a = float(x)
        b = int(a)
    except (TypeError, ValueError):
        return False
    else:
        return a == b

def evaluate51479(word):
    try:
        arr = word.split("-")
        first= split(arr[0])
        second= split(arr[1])
        #Average([int(first),int(second)])
        return Average([int(first[len(first)-1]),int(second[0])])
        #str(first[len(first)-1])+'-'+str(second[0])
    except Exception as e:
        print("in evaluate51479 : ",str(e))
        return word

def preprocess_Labs(dfLabs, type,grp_aggr):
    df = dfLabs
    df = df.drop(['diff'],axis=1)
    df = df.drop('units', axis=1)
    df = df[df['age'].notna()]
    df = df[df['value'].notna()]
    print(df.shape)
    df = df[df.age < 90 ]
    df = df[df.age > 18 ]
    print(df.shape)
    #df = df[df['diff'] > -1 ]
    print(df.shape)
    df['marital'] = df['marital'].fillna('UNKNOWN (DEFAULT)')
    df['religion'] = df['religion'].fillna('UNOBTAINABLE')
    df['marital']=df['marital'].str.lower()
    df['ethnicity']=df['ethnicity'].str.lower()
    df['religion']=df['religion'].str.lower()
    df['fluid']=df['fluid'].str.lower()
    df['category']=df['category'].str.lower()
    df.reset_index(drop=True)
    #df = df.drop('Unnamed: 0',axis=1)
    df=df.dropna(subset=['value'])
    df['value']= np.where(df["value"] == "1+", '0.25', df["value"])
    df['value']= np.where(df["value"] == "2+", '0.50', df["value"])
    df['value']= np.where(df["value"] == "3+", '0.75', df["value"])
    df['value']= np.where(df["value"] == "/0", '0', df["value"])
    df['value']= np.where(df["value"] == "/", '0', df["value"])
    df['value']= np.where(df["value"] == "13.41.1", '13.41', df["value"])
    df['value']= np.where(df["value"] == "11-20F", '11-20', df["value"])
    df['value']= np.where(df["value"] == "CHRISTA1", '1', df["value"])
    df['value']= np.where(df["value"] == "150 IS HIGHEST MEASURED PTT", '150', df["value"])
    df = df[~df['value'].astype(str).str.contains('ERROR')]
    df = df[~df['value'].astype(str).str.contains('UNABLE')]
    df = df[~df['value'].astype(str).str.contains('VOIDED')]
    df = df[~df['value'].astype(str).str.startswith('DISREGARD')]
    df = df[~df['value'].astype(str).str.startswith('QNS')]
    df = df[~df['value'].astype(str).str.contains('VERIFIED')]
    # df = df[~df['value'].astype(str).str.endswith('/0')]
    # df = df[~df['value'].astype(str).str.endswith('/')]
    df = df[df['value'].astype(str).str.strip()!='A']
    df = df[~df['value'].astype(str).str.endswith('.')]
    df = df[~df['value'].astype(str).str.endswith(' ')]
    df = df[~df['value'].astype(str).str.endswith('-')]
    df = df[~df['value'].astype(str).str.startswith('UNABLE')]
    df = df[~df['value'].astype(str).str.startswith('NOT')]
    df = df[~df['value'].astype(str).str.startswith('Not')]
    df['value']= np.where(df["value"].str.contains("GREATER THAN FIFTY"), '51', df["value"])
    df['value']= np.where(df["value"].str.contains("GRESTER THAN 50"), '51', df["value"])
    df['value']= np.where(df["value"].str.contains("LESS THN 0.3"), '0.2', df["value"])
    df['value']= np.where(df["value"].str.contains("GREATER THAN 50,"), '51', df["value"])
    df['value']= np.where(df["value"].str.contains("GREATER TAH 50,"), '51', df["value"])
    df['value']= np.where(df["value"].str.contains("LESS THAN 7.0,"), '6.9', df["value"])
    df['value']= np.where(df["value"] == "55.5 NOTIFIED ANN S. @1:00PM", '55.5', df["value"])
    df['value']= np.where(df["value"] == "0-2,TRANS", '0-2', df["value"])
    df['value']= np.where(df["value"] == "O-2", '0-2', df["value"])
    df['value']= np.where(df["value"] == "<1/HPF", '<1', df["value"])
    df['value']= np.where(df["value"] == "<1 /HPF", '<1', df["value"])
    df['value']= np.where(df["value"] == "0.", '0', df["value"])
    df['value']= np.where(df["value"] == "0-", '0', df["value"])
    df['value']= np.where(df["value"] == "2 (COARSE)", '2', df["value"])
    df['value']= np.where(df["value"] == "2 FINE GRANULAR CASTS", '2', df["value"])
    df['value']= np.where(df["value"] == "2 COARSE GRANULAR CASTS", '2', df["value"])
    df['value']= np.where(df["value"] == "20 COARSE GRANULAR CASTS", '20', df["value"])
    df['value']= np.where(df["value"] == "7.377.35", '7.37', df["value"])
    df = df[~df['value'].astype(str).str.contains('specimen lipemic')]
    df = df[~df['value'].astype(str).str.contains('SPEC.CLOTTED')]
    df = df[~df['value'].astype(str).str.contains('HEMOLYZED, SLIGHTLY')]
    df = df[~df['value'].astype(str).str.contains('ICTERIC')]
    df = df[~df['value'].astype(str).str.contains('UNNABLE TO QUANTITATE')]

    
    


    unq_vals= df.value.unique()
    

    str_unq_val=[]
    for x in unq_vals:
        #print(type(x))
        if isint(x) or isfloat(x):
            continue
        else:
            #print(type(x))
            str_unq_val.append(x)
    print(len(str_unq_val))

    for x in str_unq_val:
        new_val = evaluted_val(x)
        if new_val is not None:
            df['value']= np.where(df["value"] == x, new_val, df["value"])

    unq_labs = df.lab_id.unique()
    unq_fluid = df.fluid.unique()
    df = df.reset_index(drop=True)
    df_labs_grp = df.groupby(['lab_id'])
    final_lst=[]
    label_encoder = preprocessing.LabelEncoder() 
    newf = pd.DataFrame(columns=df.columns)
    for lab in unq_labs:
        try:
            obj= {}
            #print(df_grp.get_group((start,end)))
            plt_result = df_labs_grp.get_group((lab))
            plt_result['value'] = plt_result['value'].astype(float)
            # scaler = MinMaxScaler()
            # plt_result['value'] = scaler.fit_transform(plt_result[['value']].to_numpy())
            newf = newf.append(plt_result, ignore_index=True)
            #final_lst.append(plt_result.to_dict())
        
        except Exception as e:
            try:
                plt_result = df_labs_grp.get_group((lab))
                # print(str(e))
                # print(plt_result['hadm_id'])
                # print(plt_result['lab_id'])
                # print(plt_result['value'].sort_values(['value'],ascending=True))
                if lab == 51478:
                    plt_result['value']= np.where(plt_result["value"] == "NEG", '90', plt_result["value"])
                    plt_result['value']= np.where(plt_result["value"] == "N", '90', plt_result["value"])
                    plt_result['value']= np.where(plt_result["value"] == "TR", '100', plt_result["value"])
                    plt_result['value'] = plt_result['value'].astype(float)
                    scaler = MinMaxScaler()
                    plt_result['value'] = scaler.fit_transform(plt_result[['value']].to_numpy())
                    newf = newf.append(plt_result, ignore_index=True)
                elif lab == 51479:
                    plt_result['value'] =plt_result['value'].map(lambda y: evaluate51479(str(y)))
                    plt_result['value'] = plt_result['value'].astype(float)
                    scaler = MinMaxScaler()
                    plt_result['value'] = scaler.fit_transform(plt_result[['value']].to_numpy())
                    newf = newf.append(plt_result, ignore_index=True)
                else:
                    plt_result['value'] = plt_result['value'].astype(str)
                    # plt_result= plt_result[pd.to_numeric(plt_result['value'], errors='coerce').notnull()]
                    plt_result['value'] =label_encoder.fit_transform(plt_result['value'])
                    # enc = plt_result.groupby('value')['label'].mean().reset_index()
                    # enc = TargetEncoder(cols=['value'])
                    # fe = plt_result.groupby("value")['label'].mean()
                    # # # # #fe_ = fe/len(plt_result)
                    # plt_result['value'] =plt_result["value"].map(fe).round(4)
                    # plt_result['value'] = plt_result['value'].astype(float)
                    scaler = MinMaxScaler()
                    plt_result['value'] = scaler.fit_transform(plt_result[['value']].to_numpy())
                    # plt_result['value'] =plt_result['value'].map(lambda y: label_encoder.fit_transform([y])[0] if type(y)==str else y)
                    #plt_result = df_labs_grp.get_group((lab))
                    newf = newf.append(plt_result, ignore_index=True)
                #continue
            except Exception as e:
                print("Exception at appending:",str(e),str(lab))
                plt_result= plt_result[pd.to_numeric(plt_result['value'], errors='coerce').notnull()]
                # scaler = MinMaxScaler()
                # plt_result['value'] = scaler.fit_transform(plt_result[['value']].to_numpy())
                newf = newf.append(plt_result, ignore_index=True)
                continue

    df = newf
    #newf = pd.DataFrame(final_lst)
    #print("Before Pivot:", df)

    if type=='ML':
        df,code_dict = mergeLabstoAdmissions(df)
        #df = grp_labs(df,grp_aggr)

        # newf = df.pivot_table(
        #     values='value', 
        #     index=['label', 'marital', 'ethnicity', 'religion',
        # 'fluid', 'category', 'hadm_id', 'gender', 'age','adm_id','lab_name'], #diff
        #     columns='lab_id', 
        #     aggfunc=np.sum)
  
           
    #newf = pd.pivot_table(df, values='value', index=['label', 'marital', 'ethnicity', 'religion','fluid', 'category', 'hadm_id','gender', 'diff', 'age','adm_id'],columns=['lab_name'], aggfunc=np.sum)
        df.rename(columns=code_dict,inplace=True)
        #print("After Pivot:", df)
    
        # newdf.reset_index(drop=True)
        # df = newf
    
    # df['gender']= label_encoder.fit_transform(df['gender']) 
    # df['marital']= label_encoder.fit_transform(df['marital']) 
    # df['ethnicity']= label_encoder.fit_transform(df['ethnicity']) 
    # df['religion']= label_encoder.fit_transform(df['religion'])
    # df = pd.concat([df, pd.get_dummies(df['gender'],prefix='gender',drop_first=True)], axis=1, join='inner')
    # df = pd.concat([df, pd.get_dummies(df['marital'],prefix='marital',drop_first=True)], axis=1, join='inner')
    # df = pd.concat([df, pd.get_dummies(df['ethnicity'],prefix='ethnicity',drop_first=True)], axis=1, join='inner')
    # df = pd.concat([df, pd.get_dummies(df['religion'],prefix='religion',drop_first=True)], axis=1, join='inner')
    
   
    if type=='Graph':
        df['fluid']= label_encoder.fit_transform(df['fluid']) 
        df['category']= label_encoder.fit_transform(df['category'])
    #df['lab_name_read']= df['lab_name']
    #df['lab_name']= label_encoder.fit_transform(df['lab_name'])
    #df=df.drop('Unnamed: 10',axis=1)
    if type=='ML':
        for x in df.columns:
            #print(x)
            #print(df[x].unique())
            if x != 'hadm_id' and x!='adm_id':
                try:
                    df[x] = df[x].astype(float)
                except Exception as e:
                    print(str(e))
                    df[x] = df[x].map(lambda y: label_encoder.fit_transform([y])[0] if type(y)==str else y)
        df.columns = df.columns.str.replace('[^\w\s]', '_')
        df = df.loc[:,~df.columns.duplicated()].copy()

    return df

def evaluate_drugs(df):
    if df.shape[0]>0:
        unq_vals= df.dosage_val.unique()
        

        str_unq_val=[]
        for x in unq_vals:
            #print(type(x))
            if isint(x) or isfloat(x):
                continue
            else:
                #print(type(x))
                str_unq_val.append(x)
        print(len(str_unq_val))

        for x in str_unq_val:
            new_val = evaluted_val(x)
            if new_val is not None:
                df['dosage_val']= np.where(df["dosage_val"] == x, new_val, df["dosage_val"])
        return df
    else:
        return df


def mergeLabstoAdmissions(df_inner):
    df_inner['value']= df_inner['value'].astype(float)
    code_dict = pd.Series(df_inner.lab_name.values,index=df_inner.lab_id).to_dict()
    req_df = df_inner.drop(['adm_id','label', 'marital','lab_name', 'ethnicity', 'religion',
       'fluid', 'category', 'gender', 'age'], axis=1) #had 'adm_id' removed for testing node data
    #req_df = req_df.reset_index(drop=True)
    df_pivot = req_df.pivot_table(
        values='value', 
        index=['hadm_id'], 
        columns='lab_id', 
        aggfunc={'value':np.max})
    df_pivot['hadm_id'] = df_pivot.index
    df_pivot= df_pivot.reset_index(drop=True)
    #df=df.drop(['adm_id','lab_name','lab_id','value'], axis=1)
    unq_hadm= df_inner.hadm_id.unique()
    unq_fluids= df_inner.fluid.unique()
    unq_cat= df_inner.category.unique()
    df_grp = df_inner.groupby(['hadm_id'])
    #newdf= pd.DataFrame(columns=unq_labs)

    for x in unq_hadm:
        try:
            plt_res= df_grp.get_group(x)
            idx = df_pivot.index[df_pivot['hadm_id'] == x][0]
            df_pivot.at[idx,'adm_id']= plt_res['adm_id'].iloc[0]
            df_pivot.at[idx,'marital']= plt_res['marital'].iloc[0]
            df_pivot.at[idx,'ethnicity']= plt_res['ethnicity'].iloc[0]
            df_pivot.at[idx,'religion']= plt_res['religion'].iloc[0]
            df_pivot.at[idx,'gender']= plt_res['gender'].iloc[0]
            df_pivot.at[idx,'age']= plt_res['age'].iloc[0]
            df_pivot.at[idx,'label']= plt_res['label'].iloc[0]
            # for i in unq_fluids:
            #     my_dict = plt_res['fluid'].value_counts().to_dict()
            #     df_pivot.at[idx,'F_'+str(i)]= my_dict.get(i) if my_dict.get(i) is not None else 0
            # for i in unq_cat:
            #     my_dict = plt_res['category'].value_counts().to_dict()
            #     df_pivot.at[idx,'C_'+str(i)]= my_dict.get(i) if my_dict.get(i) is not None else 0
        except Exception as e:
            print("in exception",e)
        #break
    #print(df_pivot)
    return df_pivot,code_dict

def grp_labs(df,grp_arr):
    
    df['value'] = pd.to_numeric(df['value'], errors='coerce',downcast='float')
    df_grp = df.groupby(['adm_id','lab_id']).agg({'value':grp_arr,'label':'first','marital':'first','ethnicity':'first','religion':'first','fluid':'first','lab_id':'first','category':'first','adm_id':'first','gender':'first','hadm_id':'first','lab_name':'first','age':'first','diff':'first','units':'first'})  #'diff':'first',
    
    #df_grp = df.groupby(['adm_id','itemid'], as_index=False).sum()

    # unq_adm = df['adm_id'].unique()
    # unq_itm = df['itemid'].unique()

    # for i, adm in enumerate(unq_adm):
    #     for itm in unq_itm:
    #         try:
    #             df_result = df_grp.get_group((adm,itm))
    #             print(df_result['value'].mean())
    #          except Exception as e:
    #             print(str(e))
    #             continue

    return df_grp

def getData():
    df_lab = df_drug = df_admission = df_diagnosis = df_vitals = df_demo = df_output = pd.DataFrame()
    conn = Connection()
    diagnosis_query="""MATCH (n:D_ICD_Diagnoses) where (n.long_title contains 'sepsis') or (n.long_title contains 'septicemia')  RETURN collect(n.icd9_code) as icd_arr""" 
    df_icd9 = conn.fetch_data(diagnosis_query)
    icd9_arr= df_icd9['icd_arr'][0]
    
    #subject_query= """MATCH (n:Admissions)-[r:DIAGNOSED]-(m:D_ICD_Diagnoses) where m.icd9_code in """+str(icd9_arr)+""" RETURN collect(distinct n.subject_id) as cols"""

    #hadm_query ="""MATCH (n:Note_Events) where n.category='Discharge summary' and ((toLower(n.text) contains toLower('sepsis')) or ((toLower(n.text) contains toLower('septic shock')) or ((toLower(n.text) contains toLower('severe sepsis')))))  RETURN collect(distinct n.hadm_id) as cols"""
    subject_query= """MATCH (n:Note_Events) where n.category='Discharge summary' and (toLower(n.text) contains 'sepsis' or toLower(n.text) contains 'septic' or toLower(n.text) contains 'septicemia') RETURN collect(distinct n.subject_id) as cols"""

    df_subject = conn.fetch_data(subject_query)
    subj_arr=  df_subject['cols'][0] #df_hadm['hadm'].tolist()  #

    get_adm_query= """MATCH (n:Admissions) where n.subject_id in """+str(subj_arr)+ """ return n.subject_id as patientId, n.hadm_id as adm_id, n.admittime as time order by n.admittime asc"""

    df_hadm = conn.fetch_data(get_adm_query)
    hadm_arr = df_hadm.groupby(['patientId']).first()['adm_id'].tolist()
    print('Total number of Admission (only first adm)'+ str(len(hadm_arr)))

    #hadm_arr = hadm_arr[0:500]

   

    #print(hadm_arr)

    adm_pat_query = """MATCH (n:Admissions) where n.hadm_id in """+str(hadm_arr)+""" RETURN n.subject_id as patients, n.hospital_expire_flag as expire, n.hadm_id as hadm_id"""

    

    df_pat = conn.fetch_data(adm_pat_query)
    df_grp = df_pat.groupby(['expire'])


    # temp_lst =[]
    # for x in [0,1]:
    #     if x<=0:
    #         grp = df_grp.get_group(x)
    #         temp_lst.extend(grp['hadm_id'].values.tolist()[0:num_of_neg_samples])
    #     else:
    #         grp = df_grp.get_group(x)
    #         temp_lst.extend(grp['hadm_id'].values.tolist()[0:num_of_pos_samples])
    # for x in [0,1]:
    #     grp = df_grp.get_group(x)
    #     temp_lst.extend(grp['hadm_id'].values.tolist()[0:1882])
        
    # hadm_arr = temp_lst


    print(len(hadm_arr))
    pat_arr= df_pat['patients'].tolist()

    pat_dat_query = """ MATCH (n:Patients) where n.subject_id in """+str(hadm_arr)+""" RETURN n.gender as gender, n.dob as birth, n.subject_id as patient_id"""

    #df_pat_data = conn.fetch_data(pat_dat_query)

   

    adm_query= """MATCH (x:Patients)-[xr:ADMITTED]-(n:Admissions) where n.hadm_id in """+str(hadm_arr)+""" and x.subject_id in """+str(pat_arr)+""" RETURN n.subject_id as patients, n.hospital_expire_flag as label, n.marital_status as marital, n.ethnicity as ethnicity, n.religion as religion, n.hadm_id as hadm_id, x.gender as gender,duration.inSeconds(datetime(x.dob), datetime(n.admittime)).hours/8760 as age, apoc.node.degree(n, "HAS_LAB_EVENTS") AS output,  n.admission_type as adm_typ"""

    df_admission = conn.fetch_data(adm_query)
    

    # weights : and m.itemid in [50983,51221,50971,51249,51006,51265,50902,51301,50882,51250,50931,50912,51222,51279,51277,50868,51248,50960,50970,51237,51274,50893,51275,50804,50820,50821,50813,50818,50802]

   
    # #and m.itemid in [51275,51301,50970,50813,51265,50931,51279,51006,50893,51277]
    # #and m.itemid in [50813,50931,50912,50868,50983,51237,51006,50885,50960,50902,50971,50820,50825]
    # # high variance : [50813,50868,50885]
    # #low variance: [50960,50971,50983]
    # # handcrafted SAPS2 :  and m.itemid in [51006,50971,50882,50885,51300]
    # #and m.itemid in [51006,51237,50983,50912,50868,50813] -- for reproducing paper
    ### and m.itemid in [50882,51249,51248,50902,51222,50983,50862,50893,51301,50960] top-10

    # ## fo saps2 final: and m.itemid in [51006,50983,50833,50971,50882,51464,51516]
    lab_query= """MATCH (x:Patients)-[xr:ADMITTED]-(n:Admissions)-[r:HAS_LAB_EVENTS]->(m:D_Lab_Items) where n.hadm_id in"""+str(hadm_arr)+""" and x.subject_id in """+str(pat_arr)+""" RETURN  n.hospital_expire_flag as label,  r.value as value, r.valueUOM as units, n.marital_status as marital, n.ethnicity as ethnicity, n.religion as religion, m.fluid as fluid,  m.category as category, m.label as lab_name, n.hadm_id as adm_id,n.hadm_id as hadm_id, x.gender as gender,round(duration.inSeconds(datetime(n.admittime), datetime(r.charttime)).seconds*1.0/(60*60),2) as diff, duration.inSeconds(datetime(x.dob), datetime(n.admittime)).hours/8760 as age, m.itemid as lab_id, n.admission_type as adm_typ order by r.charttime desc""" 

    # # ## and duration.inSeconds(datetime(n.admittime), datetime(r.charttime)).hours<= 24*1 -- for 1st 24hrs

    # # # # # # # #CASE WHEN n.hospital_expire_flag>0 and duration.inSeconds(datetime(n.deathtime), datetime(r.chartdate)).seconds>0 THEN 1 ELSE 0 END AS label
    # # # # # # # #print(lab_query)

    df_lab = conn.fetch_data(lab_query)
    df_lab = df_lab.groupby(['hadm_id','lab_id']).first().reset_index()
    
    df_diagnosis_query = """MATCH (n:Admissions)-[r:DIAGNOSED]->(m:D_ICD_Diagnoses) where n.hadm_id in """+str(hadm_arr)+""" RETURN n.hadm_id as hadm_id, n.hospital_expire_flag as expire, m.long_title as title,  m.UMLSBertEmb as Embeddings"""

    df_diagnosis = conn.fetch_data(df_diagnosis_query)
    

    drug_query= """MATCH (n:Admissions)-[r:PRESCRIBED]->(m:DRUGS) where n.hadm_id in """+str(hadm_arr)+""" and r.dosage_val<>'' RETURN  ID(n) as start, ID(m) as end, r.STARTDATE as drug_start_date, r.ENDDATE as drug_end_date, r.dosage_val as dosage_val, r.dosage_unit as dosage_unit, r.generic_name as generic_name, m.name as drug_name, m.form_unit as form_unit, m.form_val as form_val, n.hadm_id as hadm_id,n.hadm_id as adm_id,n.hospital_expire_flag as label"""
    # duration.inSeconds(datetime(r.STARTDATE), datetime(r.ENDDATE)).hours as drug_duration,
    #print(drug_query)

    df_drug =  conn.fetch_data(drug_query)

    # # # # # SAPS2 -[618,51,114,198,646,677] 
    # # # # #and m.itemid in [44706,51,646] -- for reproducing paper
    # # # # # and m.itemid in [211, 220045,51,677,198,7541,40055,226559]
    # ##### top-10 [211,220045,220277,220179,455,3420,677,470,646,198]
    vitals_query = """MATCH (n:Admissions)-[r:HAS_CHART_EVNT]->(m:D_Items) where n.hadm_id in """+str(hadm_arr)+""" and r.value<>'' RETURN n.hadm_id as hadm_id,n.hadm_id as adm_id, m.label as name, m.itemid as itemid,r.value as value,r.valueUOM as UOM,n.hospital_expire_flag as label,r.charttime as time order by r.charttime desc"""

    # # # # and duration.inSeconds(datetime(n.admittime), datetime(r.charttime)).hours<= 24*1  -- for 1st 24 hours

    df_vitals =  conn.fetch_data(vitals_query)
    df_vitals = df_vitals.groupby(['hadm_id','itemid']).first().reset_index()

    # # # # # # [40055,226559] -- urine output (using foley catheter)
    # # # # # # and m.itemid in [40055,226559] -- for reproducing paper
    output_query = """MATCH (n:Admissions)-[r:HAS_OUT_EVNT]->(m:D_Items) where n.hadm_id in """+str(hadm_arr)+""" and m.itemid in [40055,226559] and r.value<>'' RETURN n.hadm_id as hadm_id,n.hadm_id as adm_id, m.label as name, m.itemid as itemid,r.value as value,r.valueUOM as UOM,n.hospital_expire_flag as label, r.charttime as time order by r.charttime desc"""

    # # and duration.inSeconds(datetime(n.admittime), datetime(r.charttime)).hours<= 24*1

    df_output =  conn.fetch_data(output_query)
    df_output = df_output.groupby(['hadm_id','itemid']).first().reset_index()

    same_demo_query = """MATCH (n:Admissions)-[r:SAME_DEMOGRAPH]->(m:Admissions) where n.hadm_id in """+str(hadm_arr)+""" and m.hadm_id in """+str(hadm_arr)+""" RETURN n.hadm_id as start, m.hadm_id as end, n.admission_type as atype, n.hadm_id as hadm_id"""

    df_demo =  conn.fetch_data(same_demo_query)

    return df_lab, df_drug, df_admission, df_diagnosis,df_vitals,df_demo,df_output

def preprocess_drugs(df):
    unq_drugs = df.drug_name.unique()
    df_drugs_grp = df.groupby(['drug_name'])
    final_lst=[]
    label_encoder = preprocessing.LabelEncoder() 
    newf = pd.DataFrame(columns=df.columns)
    for drug in unq_drugs:
        try:
            obj= {}
            plt_result = df_drugs_grp.get_group((drug))
            plt_result = plt_result[plt_result['dosage_val'].notna()]
            plt_result['form_val'] = plt_result['form_val'].fillna(0)
            plt_result['dosage_val'] = plt_result['dosage_val'].astype(float)
            # scaler = MinMaxScaler()
            # plt_result['dosage_val'] = scaler.fit_transform(plt_result[['dosage_val']].to_numpy())
            # plt_result['form_val'] = scaler.fit_transform(plt_result[['form_val']].to_numpy())
            newf = newf.append(plt_result, ignore_index=True)
        except Exception as e:
            try:
                print('exception in drug preprocessing',e)
                # fe = plt_result.groupby("dosage_val")['label'].mean()
                # # #fe_ = fe/len(plt_result)
                # plt_result['dosage_val'] =plt_result["dosage_val"].map(fe).round(4)
                # plt_result['dosage_val'] = plt_result['dosage_val'].astype(float)
            except Exception as e:
                print('exception in drug preprocessing',e)
    return newf

def mergeVitals(req_df,grp_aggr):
    df_pivot = req_df.pivot_table(
        values='value', 
        index=['hadm_id'], 
        columns='itemid', 
        aggfunc={'value':np.max})
    df_pivot['hadm_id'] = df_pivot.index
    df_pivot= df_pivot.reset_index(drop=True)
    return df_pivot

def preprocess_vitals(df,type,grp_aggr):
    code_dict = pd.Series(df.name.values,index=df.itemid).to_dict()
    unq_vitals = df.itemid.unique()
    df_vital_grp = df.groupby(['itemid'])
    final_lst=[]
    label_encoder = preprocessing.LabelEncoder() 
    newf = pd.DataFrame(columns=df.columns)

    for vital in unq_vitals:
        try:
            obj= {}
            plt_result = df_vital_grp.get_group((vital))
            plt_result = plt_result[plt_result['value'].notna()]
            plt_result['value'] = plt_result['value'].astype(float)
            # scaler = MinMaxScaler()
            # plt_result['value'] = scaler.fit_transform(plt_result[['value']].to_numpy())
            newf = newf.append(plt_result, ignore_index=True)
        except Exception as e:
            try:
                print('exception in vitals preprocessing',e)
                # fe = plt_result.groupby("value")['label'].mean()
                # # #fe_ = fe/len(plt_result)
                # plt_result['value'] =plt_result["value"].map(fe).round(4)
                # plt_result['value'] = plt_result['value'].astype(float)
            except Exception as e:
                print('exception in vitals preprocessing',e)
    if type == "ML":
        newf = mergeVitals(newf,grp_aggr)
        
    return newf,code_dict

def sentence_emd(sent):
    inputs = coder_tokenizer(sent,padding=True, truncation=True, max_length = 200, return_tensors='pt')
    sent_embed = np.mean(coder_model(**inputs).last_hidden_state[0].detach().numpy(), axis=0)
    return sent_embed


   


def map_edge_list(lst1):
    final_lst=[]
    set1= set(lst1)

    i=0
    lst1_new={}
    for val in set1:
        lst1_new[val]=i
        i=i+1
    
    return lst1_new

def convertLabs(df):
    
    # df,code_dict = mergeLabstoAdmissions(df)
    # df.rename(columns=code_dict,inplace=True)
    # df_nan = pd.DataFrame()
    # df_nan['val']=df.notna().sum()
    # df_nan.sort_values(by='val', ascending=True)
    # nan_dict= df_nan[df_nan['val']> df.shape[0]-(df.shape[0]*0.1)].to_dict()
    # nan_list= list(nan_dict['val'].keys())
    # df= df.iloc[:,df.columns.isin(nan_list)]
    # df_columns= df.columns
    # imp = SimpleImputer(strategy='median')

    # # Feeding resulting numpy array from fit_transform directly to new df2
    # df2 = pd.DataFrame(imp.fit_transform(df), columns = df_columns)
    # #df2 = df2.drop(['Unnamed: 0'], axis=1)

    unq_labs = df.lab_id.unique()
    unq_pats = df.hadm_id.unique()
    df_labs_grp = df.groupby(['hadm_id','lab_id'])

    arr_obj = []
    for p in unq_pats:
        for l in unq_labs:
            try:
                grp_res = df_labs_grp.get_group((p,l))
                #grp_res['diff'] = grp_res['diff'].apply(np.floor)
                val_lst = grp_res['value'].tolist()
                
                obj ={
                    "hadm_id": p,
                    "lab_id":l,
                    "label":grp_res['label'].tolist()[0],
                    "marital":grp_res['marital'].tolist()[0],
                    "ethnicity":grp_res['ethnicity'].tolist()[0],
                    "religion":grp_res['religion'].tolist()[0],
                    "fluid":grp_res['fluid'].tolist()[0],
                    "category":grp_res['category'].tolist()[0],
                    "adm_id":grp_res['adm_id'].tolist()[0],
                    "gender":grp_res['gender'].tolist()[0],
                    "age":grp_res['age'].tolist()[0],
                    "adm_typ":grp_res['adm_typ'].tolist()[0],
                    "lab_name":grp_res['lab_name'].tolist()[0]
                }
                
                for i,x in enumerate(val_lst):
                    col_nam= 'T'+str(i)
                    obj[col_nam] = float(x)
                #print(obj)
                arr_obj.append(obj)
            except Exception as e:
                continue
                #print('Exception in convert_labs',e)
    newdf = pd.DataFrame(arr_obj)
    

    return newdf

def getLabUMLS(df):

    conn = Connection()
    query= """Match (n:D_Lab_Items)  where n.itemid in """ +str(df['lab_id'].unique().tolist())+ """ return n.itemid as lab_id, n.UMLSBertEmb as Embeddings"""
    df_lsb_embs =  conn.fetch_data(query)
    return df_lsb_embs

def getDrugUMLS(df):
    conn = Connection()
    query= """MATCH (n:DRUGS) where n.name in """ +str(df['drug_name'].unique().tolist())+ """ return n.name as drug_name, n.UMLSBertEmb as Embeddings"""
    df_drug_embs =  conn.fetch_data(query)
    return df_drug_embs

def getVitalsUMLS(df):
    conn = Connection()
    query= """MATCH (n:D_Items) where n.itemid in """ +str(df['itemid'].unique().tolist())+ """ return n.itemid as itemid, n.UMLSBertEmb as Embeddings"""
    df_vitals_embs =  conn.fetch_data(query)
    return df_vitals_embs


def equalizeImbalance(df_adm):
    temp_lst=[]
    num_dead_pat = df_adm['label'].value_counts()[1]
    print("number of dead patients: ", num_dead_pat)
    df_grp = df_adm.groupby(['label'])
    for x in [0,1]:
        grp = df_grp.get_group(x)
        temp_lst.extend(grp['hadm_id'].values.tolist()[0:num_dead_pat])
    return temp_lst
    

def getPreprocessData(type,edge_merge,grp_aggr):
    st_time_nodes = time.time()
    df_diagnosis_features = pd.DataFrame()
    df_drugs_features = pd.DataFrame()
    df_labs_features = pd.DataFrame()
    df_vitals_features = pd.DataFrame()
    df_output_features = pd.DataFrame()
    df_demo_features = pd.DataFrame()
    df_labs, df_drugs, df_admission, df_diagnosis,df_vitals,df_demo, df_output = getData()
    

    if edge_merge=='EMerge':
        # enable to average the edges
        df_labs = grp_labs(df_labs,grp_aggr)



    end_time = time.time()

    print("Time for fetching data: ",end_time-st_time_nodes)


    if df_labs.shape[0]>0:
        df_labs = preprocess_Labs(df_labs,type,grp_aggr)
    if df_drugs.shape[0]>0:
        df_drugs = evaluate_drugs(df_drugs)
    if df_vitals.shape[0]>0:
        df_vitals,code_dict = preprocess_vitals(df_vitals,type,grp_aggr)
        df_vitals.rename(columns=code_dict,inplace=True)
        df_vitals.columns = df_vitals.columns.str.replace('[^\w\s]', '_')
        df_vitals = df_vitals.loc[:,~df_vitals.columns.duplicated()].copy()
        
    if df_output.shape[0]>0:
        df_output,code_dict = preprocess_vitals(df_output,type,grp_aggr)
        df_output.rename(columns=code_dict,inplace=True)
        df_output.columns = df_output.columns.str.replace('[^\w\s]', '_')
        df_output = df_output.loc[:,~df_output.columns.duplicated()].copy()
    
    if type=='ML':
        df_labs = df_labs.reset_index(drop=True)
        unq_hadm = df_labs['hadm_id'].unique()
        print("Before:",df_admission.shape)

        df_admission = df_admission[df_admission['hadm_id'].isin(unq_hadm)]
        # df_drugs = df_drugs[df_drugs['hadm_id'].isin(unq_hadm)]
        # df_diagnosis = df_diagnosis[df_diagnosis['hadm_id'].isin(unq_hadm)]


         #edit for equalizining class imbalance
        # eql_lst = equalizeImbalance(df_admission)
        # df_admission = df_admission[df_admission['hadm_id'].isin(eql_lst)]
        # df_labs = df_labs[df_labs['hadm_id'].isin(eql_lst)]
        # df_labs = df_labs.reset_index(drop=True)
        # df_vitals = df_vitals[df_vitals['hadm_id'].isin(eql_lst)]
        # df_vitals = df_vitals.reset_index(drop=True)
        # df_output = df_output[df_output['hadm_id'].isin(eql_lst)]
        # df_output = df_output.reset_index(drop=True)


        #df_labs = df_labs.fillna(0)
        print("Patient label count: ",df_labs['label'].value_counts())
        return df_admission, df_labs, df_vitals, df_output
    else:
        # if df_labs.shape[0]>0:
        #     df_labs = df_labs.reset_index(drop=True)
        #     unq_hadm = df_labs['hadm_id'].unique()
        # if df_drugs.shape[0]>0:
        #     df_drugs = df_drugs.reset_index(drop=True)
        #     unq_hadm = df_drugs['hadm_id'].unique()
        # if df_diagnosis.shape[0]>0:
        #     df_diagnosis = df_diagnosis.reset_index(drop=True)
        #     unq_hadm = df_diagnosis['hadm_id'].unique()
        # if df_output.shape[0]>0:
        #     df_output = df_output.reset_index(drop=True)
        #     unq_hadm = df_output['hadm_id'].unique()
        if df_vitals.shape[0]>0:
            df_vitals = df_vitals.reset_index(drop=True)
            unq_hadm = df_vitals['hadm_id'].unique()
        # if df_demo.shape[0]>0:
        #     df_demo = df_demo.reset_index(drop=True)
        #     unq_hadm = df_demo['hadm_id'].unique()
        
        
        print("Before:",df_admission.shape)

        if df_admission.shape[0]>0:
            df_admission = df_admission[df_admission['hadm_id'].isin(unq_hadm)]
        if df_drugs.shape[0]>0:
            df_drugs = df_drugs[df_drugs['hadm_id'].isin(unq_hadm)]
        if df_diagnosis.shape[0]>0:
            df_diagnosis = df_diagnosis[df_diagnosis['hadm_id'].isin(unq_hadm)]
        if df_vitals.shape[0]>0:
           df_vitals = df_vitals[df_vitals['hadm_id'].isin(unq_hadm)]
        if df_demo.shape[0]>0:
            df_demo = df_demo[df_demo['hadm_id'].isin(unq_hadm)]
        if df_output.shape[0]>0:
            df_output = df_output[df_output['hadm_id'].isin(unq_hadm)]
        
    

        # #edit for equalizining class imbalance
        # print("Drugs dataframe Size", df_drugs.shape)
        eql_lst = equalizeImbalance(df_admission)

        df_admission = df_admission[df_admission['hadm_id'].isin(eql_lst)]
        if df_labs.shape[0]>0:
            df_labs = df_labs[df_labs['adm_id'].isin(eql_lst)]
        if df_vitals.shape[0]>0:
            df_vitals = df_vitals[df_vitals['hadm_id'].isin(eql_lst)]
        if df_drugs.shape[0]>0:
            df_drugs = df_drugs[df_drugs['hadm_id'].isin(eql_lst)]
        if df_diagnosis.shape[0]>0:
            df_diagnosis = df_diagnosis[df_diagnosis['hadm_id'].isin(eql_lst)]
        if df_demo.shape[0]>0:
            df_demo = df_demo[df_demo['start'].isin(eql_lst)]
        if df_output.shape[0]>0:
            df_output = df_output[df_output['hadm_id'].isin(eql_lst)]

        if df_drugs.shape[0]>0:
            df_drugs['dosage_val']= pd.to_numeric(df_drugs['dosage_val'], errors='coerce', downcast='float')
            df_drugs = preprocess_drugs(df_drugs)
            df_drugs= df_drugs.reset_index(drop=True)


        df_labs = df_labs.reset_index(drop=True)
        df_demo = df_demo.reset_index(drop=True)
        df_output = df_output.reset_index(drop=True)
        df_vitals = df_vitals.reset_index(drop=True)
        df_drugs= df_drugs.reset_index(drop=True)
        df_diagnosis= df_diagnosis.reset_index(drop=True)
        print("After:",df_admission.shape)

        dict_start = map_edge_list(df_admission['hadm_id'].values.tolist())

        #vals = df_labs['adm_id'].values.tolist()

        df_admission['admmision_id'] = df_admission['hadm_id']
        label_encoder = preprocessing.LabelEncoder()

        if df_labs.shape[0]>0 and edge_merge == 'Nodes':
            df_labs = convertLabs(df_labs)
            


        if df_drugs.shape[0]>0:
            #df_drugs['Embeddings'] = df_drugs['drug_name'].apply(lambda x:  np.array(sentence_emd(x)))
            #df_drugs_features = expandEmbeddings(df_drugs)
            df_drug_UML= getDrugUMLS(df_drugs)
            df3 = pd.merge(df_drugs, df_drug_UML, left_on='drug_name', right_on='drug_name', how='left')
            df_drugs_features = expandEmbeddings(df3)

            # for one hot encoded vector of drug_names
            #df_drugs_features = pd.get_dummies(df_drugs['drug_name'],prefix='drug',drop_first=True)

            # for label encoded values of drug_names
            #df_drugs['drug_name']  = label_encoder.fit_transform(df_drugs['drug_name'])
            df_drugs['dosage_unit']  = label_encoder.fit_transform(df_drugs['dosage_unit'])
            df_drugs['form_unit']  = label_encoder.fit_transform(df_drugs['form_unit'])
           
        if df_diagnosis.shape[0]>0:
            
            #df_diagnosis['title']  = label_encoder.fit_transform(df_diagnosis['title'])
            #df_diagnosis['Embeddings'] = df_diagnosis['title'].apply(lambda x:  np.array(sentence_emd(x)))
            df_diagnosis_features = expandEmbeddings(df_diagnosis)
            #df_diagnosis_features = pd.get_dummies(df_diagnosis['title'],prefix='title',drop_first=True)

       
            
        if df_labs.shape[0]>0:
            df_labs["adm_id"]= df_labs["adm_id"].map(dict_start)
            #df_labs_features = pd.get_dummies(df_labs['lab_name'],prefix='lab',drop_first=True)
            df_labs_UML= getLabUMLS(df_labs)
            df3 = pd.merge(df_labs, df_labs_UML, left_on='lab_id', right_on='lab_id', how='left')
            df_labs_features = expandEmbeddings(df3)

        if df_vitals.shape[0]>0:
            df_vitals["hadm_id"]= df_vitals["hadm_id"].map(dict_start)
            
        if df_drugs.shape[0]>0:
            df_drugs["hadm_id"]= df_drugs["hadm_id"].map(dict_start)
        if df_diagnosis.shape[0]>0:
            df_diagnosis["hadm_id"]= df_diagnosis["hadm_id"].map(dict_start)
        if df_admission.shape[0]>0:
            df_admission["hadm_id"] = df_admission["hadm_id"].map(dict_start)
        if df_demo.shape[0]>0:
            df_demo["start"] = df_demo["start"].map(dict_start)
            df_demo["end"] = df_demo["end"].map(dict_start)
        if df_output.shape[0]>0:
            df_output["hadm_id"]= df_output["hadm_id"].map(dict_start)
            
        df_admission.index = df_admission['hadm_id']
        df_admission= df_admission.sort_index()


        if df_demo.shape[0]>0:
            df_demo.dropna(inplace=True)
            df_demo = df_demo.reset_index(drop=True)
            df_demo['index_col'] = df_demo.index
        if df_labs.shape[0]>0:
            df_labs['index_col'] = df_labs.index
        if df_vitals.shape[0]>0:
            df_vitals['index_col'] = df_vitals.index
        if df_output.shape[0]>0:
            df_output['index_col'] = df_output.index
        if df_admission.shape[0]>0:
            df_admission['index_col'] = df_admission.index
        if df_drugs.shape[0]>0:
            df_drugs['index_col'] = df_drugs.index
        if df_diagnosis.shape[0]>0:
            df_diagnosis['index_col'] = df_diagnosis.index

       
            

        if df_labs.shape[0]>0 and edge_merge != 'Nodes':
            df_labs['value'] = pd.to_numeric(df_labs['value'], errors='coerce',downcast='float')
        if df_vitals.shape[0]>0:
            df_vitals['value'] = pd.to_numeric(df_vitals['value'], errors='coerce',downcast='float')
        if df_output.shape[0]>0:
            df_output['value'] = pd.to_numeric(df_output['value'], errors='coerce',downcast='float')
        
        #df_labs['value'] = df_labs['value'].astype(float)
        if edge_merge=='EMerge' or edge_merge=='Nodes':
            df_labs_UML= getLabUMLS(df_labs)
            df3 = pd.merge(df_labs, df_labs_UML, left_on='lab_id', right_on='lab_id', how='left')
            df_labs_embs = expandEmbeddings(df3)
            df_labs = pd.concat([df3, df_labs_embs], axis=1, join='inner')

        
        if df_admission.shape[0]>0:
            # df_admission['gender']= label_encoder.fit_transform(df_admission['gender']) 
            # df_admission['marital']= label_encoder.fit_transform(df_admission['marital']) 
            # df_admission['ethnicity']= label_encoder.fit_transform(df_admission['ethnicity']) 
            # df_admission['religion']= label_encoder.fit_transform(df_admission['religion'])
            # df_admission['adm_typ']= label_encoder.fit_transform(df_admission['adm_typ'])
            df_admission = pd.concat([df_admission, pd.get_dummies(df_admission['gender'],prefix='gender',drop_first=False)], axis=1, join='inner')
            df_admission = pd.concat([df_admission, pd.get_dummies(df_admission['marital'],prefix='marital',drop_first=False)], axis=1, join='inner')
            df_admission = pd.concat([df_admission, pd.get_dummies(df_admission['ethnicity'],prefix='ethnicity',drop_first=False)], axis=1, join='inner')
            df_admission = pd.concat([df_admission, pd.get_dummies(df_admission['religion'],prefix='religion',drop_first=False)], axis=1, join='inner')
            df_admission = pd.concat([df_admission, pd.get_dummies(df_admission['adm_typ'],prefix='adm_typ',drop_first=False)], axis=1, join='inner')
           
        if df_vitals.shape[0]>0:
            
            df_vitals['name']= label_encoder.fit_transform(df_vitals['name'])
            df_vitals_UML= getVitalsUMLS(df_vitals)
            df3 = pd.merge(df_vitals, df_vitals_UML, left_on='itemid', right_on='itemid', how='left')
            df_vitals_features = expandEmbeddings(df3)
        if df_output.shape[0]>0:
            
            df_output['name']= label_encoder.fit_transform(df_output['name'])
            df_output_UML= getVitalsUMLS(df_output)
            df3 = pd.merge(df_output, df_output_UML, left_on='itemid', right_on='itemid', how='left')
            df_output_features = expandEmbeddings(df3)

        if df_demo.shape[0]>0:
            # df_demo['atype']= label_encoder.fit_transform(df_demo['atype'])
            df_demo_features = pd.concat([df_demo, pd.get_dummies(df_demo['atype'],prefix='atype',drop_first=False)], axis=1, join='inner')


        print("Patient label count: ",df_admission['label'].value_counts())
        total = (df_admission['label'].value_counts()[0]+df_admission['label'].value_counts()[1])
        weights = [df_admission['label'].value_counts()[1]/total,df_admission['label'].value_counts()[0]/total]
        print(weights)

        return df_admission,df_diagnosis,df_drugs,df_labs,df_vitals,df_diagnosis_features,df_demo,df_output, weights,df_drugs_features,df_labs_features,df_vitals_features,df_output_features,df_demo_features



# 211,51,676,645,198,44706,779,2981,618,114,646,677

# [51,52,53,114,152,154,189,190,211,618,1162,1165,224167,227242,227243,224643,198,184, 454, 723,223900, 223901, 220739,646,220277,223762,676,220045,442,455,6701,220179,220050,645, 44706, 779,2981,677]

# 645, 44706, 779,2981,677

# label,hadm_id,Lactate,Anion Gap,Creatinine,Sodium,Urea Nitrogen,INR_PT_,marital,ethnicity,religion,gender,age,F_blood,C_chemistry,C_hematology,C_blood gas,Arterial BP _Systolic_,SpO2,Urine Out Foley,Foley