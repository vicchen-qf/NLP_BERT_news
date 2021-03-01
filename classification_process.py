from transformers import BertTokenizer
from transformers import BertForSequenceClassification

####讀取預訓練模型####

#要更改模型，更改此路徑
model_path='./model/model_class_order'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

####建構資料型態、建構模型####

import os
import time 
import datetime
import random
import torch
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler

def to_dataloader(tokens, masks, labels):
    dataset = TensorDataset(tokens, masks, labels)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

    return dataloader


def predict_process(model, predict_dataloader, data):
    t0 = time.time()
    print("Running Prediction...")

    #---------------If there's a GPU available...
    if torch.cuda.is_available():    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    #---------------If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    model.to(device)
    
    # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
    model.eval()

    # Tracking variables 
    num_label, tmp_correct_predict =0,0
    total_predict_label = np.array([], dtype='int8')
    
    for batch in predict_dataloader:
        
        # Add batch to GPU
        # batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        # b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = batch[0].type(torch.LongTensor)
        b_input_mask = batch[1].type(torch.LongTensor)
        b_labels = batch[2].type(torch.LongTensor)
        
        
        b_input_ids =  b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)
        b_labels = b_labels.to(device)

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            # token_type_ids is the same as the "segment ids", which differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this "model" function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
            # Get the "logits" output by the model. The "logits" are the outputvalues prior to applying an activation function like the softmax.
            logits = outputs[0]
            
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # collect all the predicted labels with numpy structure
            batch_predict_label = np.argmax(logits, axis=1).flatten()
            total_predict_label = np.concatenate((total_predict_label,batch_predict_label),axis = None)
            
            
            # Calculate the accuracy for this batch of test sentences.
            tmp_correct_predict += num_correct_predict(logits, label_ids)
        
            # Accumulate the total accuracy.
            num_label += len(label_ids)

            
    data['predict_label']=total_predict_label       

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(tmp_correct_predict/num_label))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
    print(" Prediction complete!!")

    return data

def num_correct_predict(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

####處理文字資料(token、mask、label)####

import re
import torch
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, mode, datafiles, tokenizer):
        assert mode in ['training', 'testing']
        self.length = len(datafiles)
        self.tokenizer = tokenizer
        #use titles
        self.text = datafiles['titles'].values
        self.labels = datafiles['label'].values

        # convert to Bert-pretrained tokens & Create attention masks------------
        self.tokens = []
        self.masks = []
        
        for txt in self.text:
            encoded_txt = self.tokenizer.encode_plus(
                txt, 
                add_special_tokens=True,
                truncation=True,
                max_length=512,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
                )
            self.tokens.append(encoded_txt['input_ids'])
            self.masks.append(encoded_txt['attention_mask'])
        
        # Convert to tensors----------------------------------------------------
        self.tokens = torch.cat(self.tokens, dim=0)
        self.masks = torch.cat(self.masks, dim=0)
        self.labels = torch.tensor(self.labels, dtype=torch.long)


    # Inherit from Dataset needing define __len__ and __getitem__ methods-------
    def __len__(self):
        return self.length

  
    def __getitem__(self, idx):
        token = self.tokens[idx]
        mask =self.masks[idx]
        label = self.labels[idx]
        
        return (token, mask, label)

####load data(excel file)####

#load data and change label
def load_data(path):
    test = pd.read_excel(path)
    test['label'] += 1
    return test

####main function####

def bert_classifier(data_path):
    #loas data
    test = load_data(data_path)
    #tokenize text
    testdata = NewsDataset('testing', datafiles=test, tokenizer=tokenizer)
    test_dataloader = to_dataloader(testdata.tokens, testdata.masks, testdata.labels)
    ## prediction 
    predicted_data = predict_process(model, test_dataloader, test)
    return predicted_data