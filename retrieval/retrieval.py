import json
import numpy as np
import clip
import torch
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from utils import data_utils
from utils.data_utils import *
from PIL import Image

device = "cuda" 
model, preprocess = clip.load("ViT-B/32", device)

with open('nle_data/VQA-X/vqaX_train.json', 'r') as i:
    train = json.load(i)

with open('nle_data/VQA-X/vqaX_test.json', 'r') as i:
    test = json.load(i)

with open('nle_data/VQA-X/vqaX_val.json', 'r') as i:
    val = json.load(i)

m_que = np.load('retrieval/question_features.npy')
m_exp = np.load('retrieval/exp_features.npy')
#m_ans = np.load('retrieval/answer_features.npy')
#m_img = np.load('retrieval/image_features.npy')

m_que=torch.tensor(m_que).to(device)
m_exp=torch.tensor(m_exp).to(device)
#m_img=torch.tensor(m_img).to(device)

train_key=list(train)
table={}
count=0
print("test!")
for key in test.keys():
    img_name = test[key]['image_name']
    folder = '/local_datasets/vqax/train2014/' if 'train' in img_name else '/local_datasets/vqax/val2014/'
    img_path = folder + img_name

    GT_answer=data_utils.proc_ans(test[key]['answers'])

    #image feature
    image = Image.open(img_path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(image_input)
    
    #question feature
    guery_que=test[key]['question']

    guery_que=clip.tokenize([guery_que]).to(device)
    que_feature=model.encode_text(guery_que)

    q_similarity = pairwise_cosine_similarity(que_feature,m_que)
    i_similarity = pairwise_cosine_similarity(image_feature,m_exp)

    similarity=q_similarity+i_similarity

    top_values, top_indices = torch.topk(similarity, k=20)
    top_indices=top_indices.tolist()

    retrieve_keys=[train_key[top_indice] for top_indice in top_indices[0]]

    # voting method
    # reduced the negative bias on retrieval
    ans_dict={}
    for retrieve_key in retrieve_keys:
        retrieve=train[retrieve_key]
        retrieve_answer=data_utils.proc_ans(retrieve['answers'])
        if retrieve_answer not in ans_dict.keys():
            ans_dict[retrieve_answer]=1
        else:
            ans_dict[retrieve_answer]+=1

    max_ans1 = max(ans_dict, key=ans_dict.get)
    del ans_dict[max_ans1]

    if len(ans_dict)!=0:
        max_ans2 = max(ans_dict, key=ans_dict.get)
        if max_ans2 in ['yes','no'] or ans_dict[max_ans2]<3:
            max_ans2=''

    keys=[]
    for retrieve_key in retrieve_keys:
        retrieve=train[retrieve_key]
        retrieve_answer=data_utils.proc_ans(retrieve['answers'])
        if retrieve_answer==max_ans1 or retrieve_answer==max_ans2:
            keys.append(retrieve_key)
    if len(keys)==0:
        keys=retrieve_keys
    if GT_answer==max_ans1:
        count+=1
        
    table[key]=keys

with open('retrieval/qqii20_test.json', 'w') as json_file:
    json.dump(table, json_file, indent=4)

print("train!")
train_key=list(train)
table={}
for key in train.keys():
    img_name = train[key]['image_name']
    folder = '/local_datasets/vqax/train2014/' if 'train' in img_name else '/local_datasets/vqax/val2014/'
    img_path = folder + img_name

    #image feature
    image = Image.open(img_path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(image_input)
    
    #question feature
    guery_que=train[key]['question']

    guery_que=clip.tokenize([guery_que]).to(device)
    que_feature=model.encode_text(guery_que)
    
    q_similarity = pairwise_cosine_similarity(que_feature,m_que)
    i_similarity = pairwise_cosine_similarity(image_feature,m_exp)
    similarity=q_similarity+i_similarity

    top_values, top_indices = torch.topk(similarity, k=20)
    top_indices=top_indices.tolist()

    retrieve_keys=[train_key[top_indice] for top_indice in top_indices[0]]

    # voting
    ans_dict={}
    for retrieve_key in retrieve_keys:
        retrieve=train[retrieve_key]
        retrieve_answer=data_utils.proc_ans(retrieve['answers'])
        if retrieve_answer not in ans_dict.keys():
            ans_dict[retrieve_answer]=1
        else:
            ans_dict[retrieve_answer]+=1

    max_ans1 = max(ans_dict, key=ans_dict.get)
    # voting 2nd
    del ans_dict[max_ans1]

    if len(ans_dict)!=0:
        max_ans2 = max(ans_dict, key=ans_dict.get)
        if max_ans2 in ['yes','no'] or ans_dict[max_ans2]<3:
            max_ans2=''
    
    keys=[]
    for retrieve_key in retrieve_keys:
        retrieve=train[retrieve_key]
        retrieve_answer=data_utils.proc_ans(retrieve['answers'])
        if retrieve_answer==max_ans1 or retrieve_answer==max_ans2:
            keys.append(retrieve_key)

    # query key 삭제
    if key in keys:
        keys.remove(key)

    if len(keys)==0:
        keys=retrieve_keys

    table[key]=keys

with open('retrieval/qqii20_train.json', 'w') as json_file:
    json.dump(table, json_file, indent=4)