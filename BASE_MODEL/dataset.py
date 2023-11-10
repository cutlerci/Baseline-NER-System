# Scott, Charles, and Osman 9/27/23
# the ChEMU lab series is an annual competition run by Cheminformatics Elsevier Melbourne University lab. The ChEMU 
# shared NER (Named Entity Recognition) task seeks to identify chemical compounds along with their roles in a reaction. \
# Named entity recognition is a sequence classification problem, we seek to tag a sequence of words in a sentence rather than 
# classify the sentence in some way. Entities in this dataset are named according to classes such as "reaction step" and 
# "reaction product." Our goal is to label words correctly as being members of these classes, in order to glean information 
# from sets of patents that are too large for a human to read. 



# to run the code, create a virtual environment in python 3.11+ and install with  pip install -r requirements.txt 
# Run "trainer_wandb.py" which imports from the other classes 


#sample input: a number of files containing paragraphs, each containing labeled sentences. Example excerpt: 

#O	153	154	A
#O	155	162	stirred
#O	163	171	solution
#O	172	174	of
#B-STARTING_MATERIAL	175	176	(
#I-STARTING_MATERIAL	176	177	R
#I-STARTING_MATERIAL	177	178	)
#I-STARTING_MATERIAL	178	179	-
#I-STARTING_MATERIAL	179	180	8
#I-STARTING_MATERIAL	180	181	-
#I-STARTING_MATERIAL	181	182	(
#I-STARTING_MATERIAL	182	191	benzyloxy
#I-STARTING_MATERIAL	191	192	)

# sample output: micro f1, macro f1,  and f1 for each class index. 

#Note: Our current, refactored model earns a macro f1 of .96, this example output is from an earlier version 

#│      Validate metric      │       DataLoader 0        │
#│        val_avg_f1         │    0.9244508147239685     │
#│      val_f1_class_0       │    0.9460317492485046     │
#│      val_f1_class_1       │     0.95339435338974      │
#│      val_f1_class_10      │     0.983259916305542     │
#│      val_f1_class_11      │    0.9939879775047302     │
#│      val_f1_class_12      │    0.8135592937469482     │
#│      val_f1_class_13      │    0.8769268989562988     │
#│      val_f1_class_14      │    0.9610147476196289     │
#│      val_f1_class_15      │    0.5652173757553101     │
#│      val_f1_class_16      │    0.8906823396682739     │
#│      val_f1_class_17      │    0.9580487608909607     │
#│      val_f1_class_18      │    0.9719192385673523     │
#│      val_f1_class_19      │     0.971531331539154     │
#│      val_f1_class_2       │    0.9061488509178162     │
#│      val_f1_class_20      │    0.9834087491035461     │
#│      val_f1_class_21      │    0.8847235441207886     │
#│      val_f1_class_22      │    0.9802817106246948     │
#│      val_f1_class_23      │    0.9808374643325806     │
#│      val_f1_class_24      │    0.9860473871231079     │
#│      val_f1_class_3       │    0.9314526319503784     │
#│      val_f1_class_4       │    0.8810086846351624     │
#│      val_f1_class_5       │    0.9333333373069763     │
#│      val_f1_class_6       │    0.8940290212631226     │
#│      val_f1_class_7       │    0.9774339199066162     │
#│      val_f1_class_8       │    0.9861111044883728     │
#│      val_f1_class_9       │    0.9008797407150269     │
#│         valid_f1          │    0.9643142819404602     │
#│        valid_loss         │     1.116426944732666     │
#└───────────────────────────┴───────────────────────────┘

# Description of algorithm: 

# Each preprocessing step is done separately with a train, dev, and test set 
# take in data from brat format as a dataframe, labeling individual sentences as belonging to a sentence index
# for each sentence, run each word through the BERT tokenizer one at a time. If the word is split up, extend that word's label to all subwords 
# use heuristics to create B labels instead of I labels 
# feed tokenized sentences into BERT 
# ask BERT to predict the sequence 
# check sequence against extended label list and backpropogate loss 

# Preprocessing: 

# Each preprocessing step is done separately with a train, dev, and test set 
# take in data from brat format as a dataframe, labeling individual sentences as belonging to a sentence index
# for each sentence, run each word through the BERT tokenizer one at a time. If the word is split up, extend that word's label to all subwords 
# use heuristics to create B labels instead of I labels 
# feed tokenized sentences into BERT 
# ask BERT to predict the sequence 
# check sequence against extended label list and backpropogate loss 

import os
import pandas as pd
import re
import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def dir_to_df(path):
    # Create a list of files in the DATA_PATH specified directory
    file_list = [file for file in os.listdir(path) if re.search('.*\.conll', file)]
    # Create a dataframe from the first file
    data = pd.read_csv(path + "/" + file_list[0], sep = '\t', encoding = "latin-1", header = None, skip_blank_lines = False)

    # Add a NaN row for pre-processing later
    new_row = pd.DataFrame([np.nan])
    data=pd.concat([data, new_row]).reset_index(drop = True)


    # Loop through files and add to existing dataframe
    for filename in file_list[1:]:
        # Read in Data from the file
        new_data = pd.read_csv((path + "/" + filename), sep = '\t', encoding = "latin-1", header = None, skip_blank_lines = False)
        # Create a new empty row
        new_row = pd.DataFrame([np.nan])
        # Concatenate both dataframes
        data=pd.concat([data.reset_index(drop=True), new_data.reset_index(drop=True)], ignore_index=True)
        data=pd.concat([data, new_row]).reset_index(drop = True)

    data.columns = ["Label", "Start", "End", "Word"]
    return data

def remove_boring_sentences(data):
    non_o_labels = 0
    start_index = 0
    finish_index = 0
    idx_tuples = []

    #iterate through dataset by row
    for index, row in data.iterrows():
        # if the label is not O and we are not looking at a NaN row
        if row['Label'] !=('O') and row['Start'] == row['Start']:
            #increment O values
            non_o_labels += 1
        # if we are looking at a NaN row
        if row['Start'] != row['Start']:
            # if the last sentence we saw was boring
            if non_o_labels == 0:
                # save row number of the beginning of the sentence and row number of this NaN row to remove later
                finish_index = index + 1
                idx_tuples.append([start_index, finish_index])
                # set start index
                start_index = index + 1

            else:
                non_o_labels = 0
                start_index = index + 1

    # iterate through tuples and remove ranges we found
    for idx_tuple in reversed(idx_tuples):
        data = data.drop(data.index[idx_tuple[0]:idx_tuple[1]])
    data = data.reset_index()
    return data


def add_sentence_numbers(data):
    # Use the NaN rows to assign sentence number
    sentence_increment = 1
    sentence_numbers = []
    for index in range (0, len(data), 1):
      if pd.isna(data.iloc[index]["Label"]):
        sentence_numbers.append("Sentence: " + str(sentence_increment))
        sentence_increment += 1
      else:
        sentence_numbers.append(np.nan)

    # Add a new column to the data frame that associates tokens with a sentence
    data["Sentence #"] = sentence_numbers
    # Fill in the NaN values with the correct sentence numbers
    data["Sentence #"] = data["Sentence #"].fillna(method='ffill')

    # Drop rows with NaN values
    data = data.dropna()
    return data


def map_labels(data):
    # Build the Label to Index and Index to Label mappings
    label_index = 0
    label_to_index = {}
    index_to_label = {}
    unique_labels = []
    for label in data.Label.unique():
        unique_labels.append(label)
    unique_labels.sort()
    for label in unique_labels:
      label_to_index[label] = str(label_index)
      index_to_label[str(label_index)] = label
      label_index += 1
    
    return data, label_to_index

def preprocess_data_for_BERT(sentences, label_sets, label_to_index):

    encoded_token_ids = []
    encoded_attention_masks = []
    encoded_type_ids = []
    encoded_sub_labels = []

    running_tokens = []
    running_labels = []
    max_length = 512

    for sentence, labels in zip(sentences, label_sets):
        for token, label in zip(sentence, labels):
            sub_words = tokenizer.tokenize(token)
            
            if label == "O":
                token_sub_labels = [label_to_index[label]] * len(sub_words)
            else:
                token_sub_labels = [label_to_index[label]]
                if (len(sub_words) > 1):
                    token_sub_labels.extend([label_to_index["I-"+label[2:]]] * (len(sub_words)-1))
            
            running_tokens.extend(sub_words)
            running_labels.extend(token_sub_labels)

            # If running tokens exceed max_length, process and continue
            while len(running_tokens) >= max_length:
                cur_tokens = running_tokens[:max_length]
                cur_labels = running_labels[:max_length]

                # Process current tokens
                encoding = tokenizer(cur_tokens, is_split_into_words=False, add_special_tokens=True, 
                                     padding='max_length', max_length=max_length, truncation=True)
                
                encoded_token_ids.append(encoding["input_ids"])
                encoded_attention_masks.append(encoding["attention_mask"])
                encoded_type_ids.append(encoding["token_type_ids"])
                cur_labels.extend(["-100"] * (max_length-len(cur_labels)))
                encoded_sub_labels.append(cur_labels)

                # Remove processed tokens and continue
                running_tokens = running_tokens[max_length:]
                running_labels = running_labels[max_length:]

    # Handle any remaining tokens after processing all sentences
    if running_tokens:
        encoding = tokenizer(running_tokens, is_split_into_words=False, add_special_tokens=True, 
                             padding='max_length', max_length=max_length, truncation=True)
        encoded_token_ids.append(encoding["input_ids"])
        encoded_attention_masks.append(encoding["attention_mask"])
        encoded_type_ids.append(encoding["token_type_ids"])
        running_labels.extend(["-100"] * (max_length-len(running_labels)))
        encoded_sub_labels.append(running_labels)

    return {
        "token_ids" : torch.tensor(encoded_token_ids, dtype=torch.long),
        "attention_masks" : torch.tensor(encoded_attention_masks, dtype=torch.long),
        "labels" : torch.tensor(encoded_sub_labels, dtype=torch.long)
      }

class BERTNERDataset(TensorDataset):
    def __init__(self, data_path):
        data = dir_to_df(data_path)
        data = remove_boring_sentences(data)
        data = add_sentence_numbers(data)
        #Group By Sentence Number
        data_grouped = data.groupby("Sentence #").agg({'Label':list, 'Start':list, 'End':list, 'Word': list})

        train_labels = data_grouped["Label"].tolist()
        train_sentences = data_grouped["Word"].tolist()

        data, train_to_index = map_labels(data)


        preprocessed_train_data = preprocess_data_for_BERT(train_sentences, train_labels, train_to_index)

        super(BERTNERDataset, self).__init__(preprocessed_train_data["token_ids"], preprocessed_train_data["attention_masks"], preprocessed_train_data["labels"])

# #test the dataset
# ner_dataset = BERTNERDataset('ChEMU2023_FOR_CLASS/dev')

# #print some samples
# print(ner_dataset[0])
# print(ner_dataset[1])
# print(ner_dataset[2])
