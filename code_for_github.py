import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import json
# from cococaption.pycocotools.coco import COCO
# from cococaption.pycocoevalcap.eval import COCOEvalCap
from PIL import Image
from accelerate import Accelerator
from models.BaselineGPT import GPT2LMHeadModel
from models.clip_vit import ImageEncoder
from utils import data_utils
from utils.data_utils import *
from utils.eval_utils import top_filtering
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import evaluate
from evaluate import load
import clip
import numpy as np


def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad


def load_checkpoint(ckpt_path, epoch):
    
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_0'
    filename = 'ckpt_stats_' + str(epoch) + '.tar'
    
    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path + tokenizer_name)        # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(ckpt_path + model_name).to(device)   # load model with config
    opt = torch.load(ckpt_path + filename)
    optimizer = get_optimizer(model, learning_rate)
    optimizer.load_state_dict(opt['optimizer_state_dict'])
    start_epoch = opt['epoch'] + 1
    scheduler_dic = opt['scheduler']
    del opt
    torch.cuda.empty_cache()

    return tokenizer, model, optimizer, scheduler_dic, start_epoch

def load_pretrained():
    # model_path = '/local_datasets/vqax/pretrain_model'
    # tokenizer_path = '/local_datasets/vqax/pretrain_tokenizer_0'
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)        # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(pretrain_model_path).to(device)   # load model with config
    return tokenizer, model
    

def save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path, **kwargs):
    
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_{}'.format(str(epoch))
    
    if epoch:
        tokenizer.save_pretrained(ckpt_path + tokenizer_name)   # save tokenizer
        
    unwrapped_model.save_pretrained(ckpt_path + model_name, save_function=accelerator.save)


def make_input_text(tokenizer,max_seq_len,question,answer,explanation):
        i_segment_id,r_segment_id,q_segment_id, a_segment_id, e_segment_id = tokenizer.convert_tokens_to_ids([
                                                                                    '<image>',
                                                                                    '<retrieve>',
                                                                                    '<question>', 
                                                                                    '<answer>', 
                                                                                    '<explanation>'])
        
        tokens = tokenizer.tokenize(question)
        labels = [-100] * len(tokens)   # we dont want to predict the question, set to pad to ignore in XE
        segment_ids = [q_segment_id] * len(tokens)

        answer = [tokenizer.bos_token] + tokenizer.tokenize(" the answer is " + answer)
        answer_len = len(answer)
        exp_token = tokenizer.tokenize(" because " + explanation) + [tokenizer.eos_token]
        exp_len = len(exp_token)
        tokens += answer + exp_token
        labels += [-100] + answer[1:] + exp_token   # labels will be shifted in the model, so for now set them same as tokens
        segment_ids += [a_segment_id] * answer_len
        segment_ids += [e_segment_id] * exp_len

        if len(tokens) > max_seq_len :
            tokens = tokens[:max_seq_len]
            labels = labels[:max_seq_len]
            segment_ids = segment_ids[:max_seq_len]


        assert len(tokens) == len(segment_ids) 
        assert len(tokens) == len(labels)
        
        seq_len = len(tokens)
        padding_len = max_seq_len - seq_len
        tokens = tokens + ([tokenizer.pad_token] * padding_len)
        labels = labels + ([-100] * padding_len)
        
        segment_ids += ([e_segment_id] * padding_len)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        labels = [tokenizer.convert_tokens_to_ids(t) if t!=-100 else t for t in labels]
        labels = torch.tensor(labels, dtype=torch.long)
        
        return input_ids, labels, segment_ids

def make_clipfeature(rets):
    rets = clip.tokenize(rets).to(device)
    with torch.no_grad():
        rets_feature = clipmodel.encode_text(rets)

    #retrieval_embed = torch.mean(rets_feature, dim=0).unsqueeze(0).to('cpu')
    retrieval_embed=rets_feature.to('cpu')
    return retrieval_embed

class VQAXTrainDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):
        
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len       # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())
        
        for k,v in self.data.items():
            if len(v['explanation']) > 1:   # some questions have more than one explanation
                # duplicate them for loading. -1 because one explanation is already in ids_list
                self.ids_list += [str(k)] * (len(v['explanation']) - 1)    

        self.index_tracker = {k: len(v['explanation']) - 1 for k,v in self.data.items()}
        with open('retrieval/qqii20_train.json', 'r') as table:
            table = json.load(table)
        self.table = table

        self.exp_features = np.load('retrieval/exp_features.npy')
        self.image_features = np.load('retrieval/image_features.npy')
        self.question_features = np.load('retrieval/question_features.npy')
        self.ans_features = np.load('retrieval/answer_features.npy')
        

    def __getitem__(self, i):
        
        quention_id = self.ids_list[i]
        sample = self.data[quention_id]
        img_name = sample['image_name']
        query_question = data_utils.proc_ques(sample['question'])    # question
        answer = data_utils.proc_ans(sample['answers'])

        exp_idx = self.index_tracker[quention_id]    # the index of the explanation for questions with multiple explanations
        if exp_idx > 0:
            self.index_tracker[quention_id] -= 1    # decrease usage
                
        text_b = sample['explanation'][exp_idx]   # explanation

        # tokenization process
        input_ids, labels, segment_ids = make_input_text(self.tokenizer,self.max_seq_len,query_question,answer,text_b)

        folder = '/local_datasets/vqax/train2014/' if 'train' in img_name else '/local_datasets/vqax/val2014/'
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # adapt retrieval
        keys=self.table[quention_id]
        train_key=list(self.data)

        features=[]
        exp_features_list=[]
        ans_features_list=[]
        for k in keys[:10]:
            index=train_key.index(k)
            exp_features_list.append(self.exp_features[index])
            ans_features_list.append(self.ans_features[index])
        features.append(np.mean(exp_features_list,axis=0))
        features.append(np.mean(ans_features_list,axis=0))

        features=torch.tensor(features)
        
        ret_feature=features

        qid = torch.LongTensor([int(quention_id)])
        #return (img_concat, qid, input_ids, labels, segment_ids)
        return (img, qid, input_ids, labels, segment_ids,ret_feature)

    def __len__(self):
        return len(self.ids_list)

class VQAXEvalDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):

        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len       # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())
        with open('retrieval/qqii20_test.json', 'r') as table:
            table = json.load(table)
        self.table = table
        with open('nle_data/VQA-X/vqaX_train.json', 'r') as train:
            train = json.load(train)
        self.train = train

        self.exp_features = np.load('retrieval/exp_features.npy')
        self.image_features = np.load('retrieval/image_features.npy')
        self.question_features = np.load('retrieval/question_features.npy')
        self.ans_features = np.load('retrieval/answer_features.npy')


    def __getitem__(self, i):
        
        quention_id = self.ids_list[i]
        sample = self.data[quention_id]
        img_name = sample['image_name']
        folder = '/local_datasets/vqax/train2014/' if 'train' in img_name else '/local_datasets/vqax/val2014/'
        img_path = folder + img_name
        
        query_question = data_utils.proc_ques(sample['question'])    # question
        # tokenization process
        i_segment_id,r_segment_id, q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<image>','<retrieve>','<question>', '<answer>', '<explanation>'])
        tokens = self.tokenizer.tokenize(query_question)
        segment_ids = [q_segment_id] * len(tokens)

        answer = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the answer is")
        answer_len = len(answer)
        tokens += answer 

        segment_ids += [a_segment_id] * answer_len

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)

        folder = '/local_datasets/vqax/train2014/' if 'train' in img_name else '/local_datasets/vqax/val2014/'
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # adapt retrieval
        keys=self.table[quention_id]
        train_key=list(self.train)

        features=[]
        exp_features_list=[]
        ans_features_list=[]
        for k in keys[:10]:
            index=train_key.index(k)
            exp_features_list.append(self.exp_features[index])
            ans_features_list.append(self.ans_features[index])
        features.append(np.mean(exp_features_list,axis=0))
        features.append(np.mean(ans_features_list,axis=0))

        features=torch.tensor(features)
        
        ret_feature=features

        qid = torch.LongTensor([int(quention_id)])
        
        return (img, qid, input_ids, segment_ids,ret_feature)

    def __len__(self):
        return len(self.ids_list)

def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],  
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer

def sample_sequences(model, tokenizer, loader):
    
    model.eval()
    results_exp = []
    results_full = []
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>','<image>','<retrieve>', '<question>', '<answer>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    because_token = tokenizer.convert_tokens_to_ids('Ġbecause')
    max_len = 20
    
    for i,batch in enumerate(loader):
        
        current_output = []
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, img_id, input_ids, segment_ids,ret_feature = batch
        
        img_embedding = image_encoder(img).to(device)

        always_exp = False
        with torch.no_grad():
            for step in range(max_len + 1):
                
                if step == max_len:
                    break
                outputs = model(input_ids=input_ids, 
                                past_key_values=None, 
                                attention_mask=None, 
                                token_type_ids=segment_ids, 
                                position_ids=None, 
                                encoder_hidden_states=img_embedding, 
                                encoder_attention_mask=None, 
                                labels=None, 
                                use_cache=False, 
                                return_dict=True,
                                retrieve=ret_feature,
                                )
                
                lm_logits = outputs.logits 
                logits = lm_logits[0,0, -1, :] / temperature
                logits = top_filtering(logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(logits, dim=-1)
                prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)
                
                if prev.item() in special_tokens_ids:
                    break
                
                # take care of when to start the <explanation> token
                if not always_exp:
                    
                    if prev.item() != because_token:
                        new_segment = special_tokens_ids[-2]   # answer segment
                    else:
                        new_segment = special_tokens_ids[-1]   # explanation segment
                        always_exp = True
                else:
                    new_segment = special_tokens_ids[-1]   # explanation segment
                    
                new_segment = torch.LongTensor([new_segment]).to(device)
                current_output.append(prev.item())
                input_ids = torch.cat((input_ids, prev.unsqueeze(0).unsqueeze(0)), dim = 2)
                segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0).unsqueeze(0)), dim = 2)
        
        
        decoded_sequences = tokenizer.decode(current_output, skip_special_tokens=True).lstrip()
        results_full.append({"image_id": img_id.item(), "caption": decoded_sequences})
            
        if 'because' in decoded_sequences:
            cut_decoded_sequences = decoded_sequences.split('because')[-1].strip()
        else:
            cut_decoded_sequences = " ".join(decoded_sequences.split()[2:])
        
        results_exp.append({"image_id": img_id.item(), "caption": cut_decoded_sequences})
        print("\rEvaluation: Finished {}/{}".format(i, len(loader)), end='          ')
        print('결과는 : ',decoded_sequences)
            
    return results_full, results_exp


accelerator = Accelerator()
device = accelerator.device

finetune_pretrained = True   # if True, finetunes from the image captioning model
eval_batch_size = 1
img_size = 224
ckpt_path = '/local_datasets/vqax/'
caption_save_path = 'results/qqii/' 
nle_data_train_path = 'nle_data/VQA-X/vqaX_train.json'
nle_data_test_path = 'nle_data/VQA-X/vqaX_test.json'
nle_data_val_path = 'nle_data/VQA-X/vqaX_val.json'
pretrain_model_path = '/local_datasets/vqax/pretrain_model'
tokenizer_path = '/local_datasets/vqax/pretrain_tokenizer_0'
max_seq_len = 40
load_from_epoch = None
no_sample = True   
top_k =  0
top_p =  0.9
batch_size = 32   # per GPU 
num_train_epochs = 30
weight_decay = 0
learning_rate = 2e-5 if not finetune_pretrained else 1e-5
gradient_accumulation_steps = 1   
start_epoch = 0
temperature = 1

image_encoder = ImageEncoder(device).to(device) # "ViT-B/16"
change_requires_grad(image_encoder, False)

clipmodel, _ = clip.load("ViT-B/32", 'cuda')

if load_from_epoch is not None:
    tokenizer, model, optimizer, scheduler_dic, start_epoch = load_checkpoint(ckpt_path, load_from_epoch)
    
else:
    
    if finetune_pretrained:
        tokenizer, model = load_pretrained()
        optimizer = get_optimizer(model, learning_rate)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        orig_num_tokens = len(tokenizer.encoder)
        
        num_new_tokens = tokenizer.add_special_tokens({'pad_token': '<pad>',
                                                       'additional_special_tokens': ['<image>','<retrieve>','<question>', '<answer>', '<explanation>']})
        
        assert len(tokenizer) == orig_num_tokens + num_new_tokens

        config = AutoConfig.from_pretrained('distilgpt2')
        
        # Add configs
        setattr(config, 'img_size', None)
        setattr(config, 'max_seq_len', None)   
        config.img_size = img_size
        config.max_seq_len = max_seq_len 
        config.add_cross_attention = True
        
        model = GPT2LMHeadModel.from_pretrained('distilgpt2', config = config)
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)

        optimizer = get_optimizer(model, learning_rate)

print("Model Setup Ready...")

clip_model, clip_preprocess = clip.load('ViT-B/32', device)
img_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = VQAXTrainDataset(path = nle_data_train_path, 
                                 transform = img_transform, 
                                 tokenizer = tokenizer, 
                                 max_seq_len = max_seq_len)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = batch_size, 
                                           shuffle=True,
                                           pin_memory=True)

test_dataset = VQAXEvalDataset(path = nle_data_test_path,      
                               transform = img_transform, 
                               tokenizer = tokenizer, 
                               max_seq_len = max_seq_len)


test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size = 1, 
                                          shuffle=False, 
                                          pin_memory=True)

model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

t_total = (len(train_loader) // gradient_accumulation_steps) * num_train_epochs
warmup_steps = 0   # 0.10 * t_total
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

if load_from_epoch is not None:
    scheduler.load_state_dict(scheduler_dic)

#val_loss = 999999
for epoch in range(start_epoch, num_train_epochs):
    
    model.train()
    accum_loss = 0
    
    for step, batch in enumerate(train_loader):
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, _, input_ids, labels, segment_ids, ret_feature = batch

        img_embedding = image_encoder(img).to(device)

        outputs = model(input_ids=input_ids, 
                        past_key_values=None, 
                        attention_mask=None, 
                        token_type_ids=segment_ids, 
                        position_ids=None, 
                        encoder_hidden_states=img_embedding, 
                        encoder_attention_mask=None, 
                        labels=labels,
                        use_cache=False, 
                        return_dict=True,
                        retrieve=ret_feature
                        )
        
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        accum_loss += loss.item()
        
        if step % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accelerator.print("\rEpoch {} / {}, Iter {} / {}, Loss: {:.3f}".format(epoch, 
                                                                                   num_train_epochs, 
                                                                                   step, len(train_loader), 
                                                                                   accum_loss), 
                              end='          ')
            accum_loss = 0

    unwrapped_model = accelerator.unwrap_model(model)
    save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path)
    print('model save on epoch ',epoch)
    if epoch:
        results_full, results_exp = sample_sequences(model, tokenizer, test_loader)

        resFileExp = caption_save_path + 'captions_exp_' + str(epoch) + '.json'
        unf_resFileExp = caption_save_path + 'unf_captions_exp_' + str(epoch) + '.json'
        unf_resFileFull = caption_save_path + 'unf_captions_full_' + str(epoch) + '.json'
        save_scores_pathExp = caption_save_path + 'scores_exp_' + str(epoch) + '.json'

        with open(unf_resFileExp, 'w') as w:
            json.dump(results_exp, w)
            
        with open(unf_resFileFull, 'w') as w:
            json.dump(results_full, w)