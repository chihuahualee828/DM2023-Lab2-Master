### Read files, merge dataframe, and split train/test:
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
### training data
iden_df = pd.read_csv("../data/kaggle/data_identification.csv",
                         header=0)
emotion_df = pd.read_csv("../data/kaggle/emotion.csv",
                         header=0)
print(len(iden_df), len(emotion_df))
# Load the tweets raw data file
with open("../data/kaggle/tweets_DM.json", "r") as file:
    tweets_data = [json.loads(line) for line in file]

len(tweets_data)
# Extract relevant information from the tweets data
tweets_info = []
for tweet in tweets_data:
    tweet_id = tweet["_source"]["tweet"]["tweet_id"]
    hashtags = tweet["_source"]["tweet"]["hashtags"]
    text = tweet["_source"]["tweet"]["text"]
    tweets_info.append({"tweet_id": tweet_id, "hashtags": hashtags, "text": text})

# Create a dataframe from the tweets information
tweets_df = pd.DataFrame(tweets_info)
merged_df = pd.merge(iden_df, tweets_df, on="tweet_id")

train_df = merged_df[merged_df["identification"] == "train"]
test_df = merged_df[merged_df["identification"] == "test"]

train_df = pd.merge(train_df, emotion_df, on="tweet_id")

print(len(train_df))
print(len(test_df))
# train_df = train_df.sample(frac=0.1, replace=True, random_state=1)
# print(len(train_df))
# import sys
# sys.path.append('../helpers')
# import data_mining_helpers as dmh
# train_df.isnull().apply(lambda x: dmh.check_missing_values(x))
# test_df.isnull().apply(lambda x: dmh.check_missing_values(x))
### Roberta
import torch
from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pickle
target_list = ['anger', 'anticipation', 'disgust', 'fear', 'sadness', 'surprise', 'trust', 'joy']
# tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
# encoded_data = torch.load('encoded_data.pth')


tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
encoded_data = tokenizer(list(train_df['text']), truncation=True, padding=True, return_tensors='pt', max_length=128)


input_ids = encoded_data['input_ids'][0]

# Decode the token IDs to obtain the original sentence
decoded_sentence = tokenizer.decode(input_ids, skip_special_tokens=True)


label_map = {label: i for i, label in enumerate(target_list)}

# Map emotion labels to each tweet's tensor
label_map = {label: i for i, label in enumerate(target_list)}
encoded_data['labels'] = torch.tensor([label_map[label] for label in train_df['emotion']])

# one_hot_labels = torch.zeros((len(encoded_data['input_ids']), len(target_list)))

# # Iterate through the DataFrame and set the corresponding elements to 1
# for i, label in enumerate(encoded_data['labels']):
#     one_hot_labels[i, label] = 1
# encoded_data['labels'] = one_hot_labels

print(encoded_data['labels'].shape)
print(encoded_data['input_ids'].shape)
# Split into train and valid, 9:1
# train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(encoded_data['input_ids'],
#                                                                                     encoded_data['labels'],
#                                                                                     random_state=42,
#                                                                                     test_size=0.1)

model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=len(label_map), problem_type = 'single_label_classification')

# "cuda" if torch.cuda.is_available() else "cpu"
import gc
torch.cuda.empty_cache()
gc.collect()

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# train_dataset = TensorDataset(train_inputs, train_labels)
# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# validation_dataset = TensorDataset(validation_inputs, validation_labels)
# validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=False)

# # Define loss function
# criterion = torch.nn.CrossEntropyLoss()

# # Training loop
# num_epochs = 5  # Adjust as needed
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
class Trainer:
    def __init__(self, log_dir, model, encoded_data):
        '''Initialize the varibles for training
        Args:
            log_dir: (pathlib.Path) the direction used for logging
        '''
        self.log_dir = log_dir
        print(self.log_dir)
        # Split into train and valid, 9:1
        self.train_inputs, self.validation_inputs, self.train_attentions, self.validation_attentions, self.train_labels, self.validation_labels = train_test_split(encoded_data['input_ids'],
                                                                                            encoded_data['attention_mask'],
                                                                                            encoded_data['labels'],
                                                                                            random_state=42,
                                                                                            test_size=0.2)
        self.train_dataset = TensorDataset(self.train_inputs, self.train_attentions, self.train_labels)
        self.batch_size = 16

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.validation_dataset = TensorDataset(self.validation_inputs, self.validation_attentions, self.validation_labels)
        self.valid_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)



        # Datasets and dataloaders
        # 1. Split the whole training data into train and valid (validation)
        # 2. Make the corresponding dataloaders

        # self.train_loader = DataLoader(self.train_set, 16, shuffle=True, num_workers=0)
        # self.valid_loader = DataLoader(self.valid_set, 16, shuffle=False, num_workers=0)

        # model, loss function, optimizer
        self.device = 'cuda'
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = 7e-6
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.lr_decay = 0.1
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=self.lr_decay, last_epoch=-1, verbose=False)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        self.max_epoch = 5

    def run(self):
        training_result_dir = self.log_dir / 'training_result'
        training_result_dir.mkdir(parents=True)
        metrics = {'train_loss': [], 'valid_loss': []}
        lrs = []
        for self.epoch in range(self.max_epoch): # epochs
            train_loss = self.train() # train 1 epoch
            valid_loss = self.valid() # valid 1 epoch
            print('lr:',get_lr(self.optimizer))
            lrs.append(get_lr(self.optimizer))
            print(f'Epoch {self.epoch:03d}:')
            print('train loss:', train_loss)
            print('valid loss:', valid_loss)
            metrics['train_loss'].append(train_loss)
            metrics['valid_loss'].append(valid_loss)
            print(f"train_losses: {metrics['train_loss']}")
            print(f"valid_losses: {metrics['valid_loss']}")
            # Save the parameters(weights) of the model to disk
            if torch.tensor(metrics['valid_loss']).argmin() == self.epoch:
                print("saved")
                torch.save(self.model.state_dict(), str(training_result_dir / 'model.pth'))
        
        # Plot the loss curve against epoch
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
        ax.set_title(f'Loss(batch_size:{self.batch_size}, lr:{self.lr}, lr_decay:{self.lr_decay if self.lr_decay else "False"})')
        ax.plot(range(self.epoch + 1), metrics['train_loss'], label='Train')
        ax.plot(range(self.epoch + 1), metrics['valid_loss'], label='Valid')
        ax.legend()
        plt.show()
        fig.savefig(str(training_result_dir / 'metrics.jpg'))
        plt.close()

        # fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
        # ax.set_title('Learning rate')
        # ax.plot(range(self.epoch + 1), lrs)
        # ax.legend()
        # plt.show()
        # fig.savefig(str(training_result_dir / 'lr.jpg'))
        # plt.close()

    def train(self):
        '''Train one epoch
        1. Switch model to training mode
        2. Iterate mini-batches and do:
            a. clear gradient
            b. forward to get loss
            c. loss backward
            d. update parameters
        3. Return the average loss in this epoch
        '''
        self.model.train()
        loss_steps = []

        for batch_inputs, batch_attentions, batch_labels in tqdm(self.train_loader):
            batch_inputs, batch_attentions, batch_labels = batch_inputs.to(self.device), batch_attentions.to(self.device), batch_labels.to(self.device)
            # for each in batch_inputs:
            #     decoded_sentence = tokenizer.decode(each, skip_special_tokens=True)
            #     print("Tokenized Sentence:", decoded_sentence)
            # Forward pass
            outputs = model(input_ids=batch_inputs, attention_mask=batch_attentions ,labels=batch_labels)
            loss = outputs.loss
            # print(outputs.logits)
            # print(torch.argmax(outputs.logits, dim=1))
            # predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
            # print(predictions)
            # predictions = torch.argmax(predictions, dim=1)
            # print(batch_labels, predictions)
            # print(loss)
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_steps.append(loss.detach().item())
            # total_loss += loss.item()
        self.scheduler.step()
        
        # Calculate average training loss for the epoch
        avg_train_loss = sum(loss_steps) / len(self.train_loader)
        print(f"Epoch {self.epoch + 1}, Avg. Training Loss: {avg_train_loss:.4f}")
        
        return avg_train_loss

    @torch.no_grad()
    def valid(self):
        '''Validate one epoch
        1. Switch model to evaluation mode and turn off gradient (by @torch.no_grad() or with torch.no_grad())
        2. Iterate mini-batches and do forwarding to get loss
        3. Return average loss in this epoch
        '''
        self.model.eval()
        loss_steps = []
        all_predictions = []
        all_true_labels = []

        for batch_inputs, batch_attentions, batch_labels in self.valid_loader:
            batch_inputs, batch_attentions, batch_labels = batch_inputs.to(self.device), batch_attentions.to(self.device), batch_labels.to(self.device)
            # Forward pass
            outputs = self.model(input_ids=batch_inputs, attention_mask=batch_attentions)
            # Predictions
            predictions = outputs.logits.cpu()
            # predictions = torch.argmax(predictions, dim=1).cpu().numpy()
            # predictions = torch.nn.functional.softmax(outputs.logits).cpu()
            true_labels = batch_labels.cpu()
            # print(predictions, true_labels)
            loss_steps.append(self.criterion(predictions, true_labels))
            true_labels = true_labels.numpy()
            all_predictions.extend(predictions)
            all_true_labels.extend(true_labels)

            

        # Calculate accuracy on the validation set
        avg_valid_loss = sum(loss_steps) / len(self.valid_loader)
        print(f"Epoch {self.epoch + 1}, Validation Loss: {avg_valid_loss:.4f}")
        return avg_valid_loss

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


log_dir = Path('./runs/') / f'{datetime.now():%b%d_%H_%M_%S}'
log_dir.mkdir(parents=True, exist_ok=True)
Trainer(log_dir, model, encoded_data).run()







