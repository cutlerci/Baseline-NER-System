"""
Data Preprocessing Functions

This Python file contains a set of 6 functions designed for data preprocessing tasks. 
These functions are intended to be used to prepare and clean data for use in the ChEMUDataModule. 
Each function serves a specific purpose and contributes to the data preparation process. 

Functions:
- dir_to_df: Creates a dataframe from folder of raw data files.
- remove_only_o_sentences: Removes sentences containing only "O" class labels.
- add_sentence_numbers: Associates tokens with a sentence ID number.
- map_labels: Builds a mapping between class labels and prediction indexes.
- multi_sentence_BERT_preprocessing: Prepares data samples for use with BERT.
- single_sentence_BERT_preprocessing: Prepares data samples for use with BERT.
- run_data_preprocessing: Runs all the data preprocessing steps in order.

Please refer to the function-specific docstrings for more detailed information on their implementation and usage.

Authors: Scott, Osman, and Charles
Date: 10/19/2023
"""

import os
import re
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# These labels are grouped by tag and sorted by the occurrence of the sum of their B and I tags
SORTED_UNIQUE_LABELS_IOB = [
    'O',
    'B-REACTION_PRODUCT',
    'I-REACTION_PRODUCT',
    'B-STARTING_MATERIAL',
    'I-STARTING_MATERIAL',
    'B-OTHER_COMPOUND',
    'I-OTHER_COMPOUND',
    'B-REACTION_STEP',
    'I-REACTION_STEP',
    'B-TEMPERATURE',
    'I-TEMPERATURE',
    'B-REAGENT_CATALYST',
    'I-REAGENT_CATALYST',
    'B-YIELD_OTHER',
    'I-YIELD_OTHER',
    'B-WORKUP',
    'I-WORKUP',
    'B-YIELD_PERCENT',
    'I-YIELD_PERCENT',
    'B-TIME',
    'I-TIME',
    'B-SOLVENT',
    'I-SOLVENT',
    'B-EXAMPLE_LABEL',
    'I-EXAMPLE_LABEL'
]

SORTED_UNIQUE_LABELS = [
    'O',
    'REACTION_PRODUCT',
    'STARTING_MATERIAL',
    'OTHER_COMPOUND',
    'REACTION_STEP',
    'TEMPERATURE',
    'REAGENT_CATALYST',
    'YIELD_OTHER',
    'WORKUP',
    'YIELD_PERCENT',
    'TIME',
    'SOLVENT',
    'EXAMPLE_LABEL'
]

# Standard indices for all labels in ChEMU
LABEL_TO_INDEX = {
    'O': 0,
    'REACTION_PRODUCT': 1,
    'STARTING_MATERIAL': 2,
    'OTHER_COMPOUND': 3,
    'REACTION_STEP': 4,
    'TEMPERATURE': 5,
    'REAGENT_CATALYST': 6,
    'YIELD_OTHER': 7,
    'WORKUP': 8,
    'YIELD_PERCENT': 9,
    'TIME': 10,
    'SOLVENT': 11,
    'EXAMPLE_LABEL': 12
}

LABEL_TO_INDEX_IOB = {
    'O': 0,
    'B-REACTION_PRODUCT': 1,
    'I-REACTION_PRODUCT': 2,
    'B-STARTING_MATERIAL': 3,
    'I-STARTING_MATERIAL': 4,
    'B-OTHER_COMPOUND': 5,
    'I-OTHER_COMPOUND': 6,
    'B-REACTION_STEP': 7,
    'I-REACTION_STEP': 8,
    'B-TEMPERATURE': 9,
    'I-TEMPERATURE': 10,
    'B-REAGENT_CATALYST': 11,
    'I-REAGENT_CATALYST': 12,
    'B-YIELD_OTHER': 13,
    'I-YIELD_OTHER': 14,
    'B-WORKUP': 15,
    'I-WORKUP': 16,
    'B-YIELD_PERCENT': 17,
    'I-YIELD_PERCENT': 18,
    'B-TIME': 19,
    'I-TIME': 20,
    'B-SOLVENT': 21,
    'I-SOLVENT': 22,
    'B-EXAMPLE_LABEL': 23,
    'I-EXAMPLE_LABEL': 24
}

def dir_to_df(path):
    """
    Takes a filepath to a directory of raw datafiles and creates a Pandas dataframe.

    Function pseudocode / implementation details:
    A) Create a list of files in the raw data directory.
    B) Create a dataframe from the first file.
    C) Add a NaN row for later pre-processing steps.
    D) Loop through remaining files and extend the existing dataframe.
        E) Read in the raw data from the file.
        F) Create a a NaN row for later.
        G) Extend the existing dataframe.
    H) Add column headers to the dataframe.
    """
    file_list = [file for file in os.listdir(path) if re.search('.*\.conll', file)] # A
    data = pd.read_csv(path + "/" + file_list[0], sep = '\t', encoding = "latin-1", header = None, skip_blank_lines = False) # B
    new_row = pd.DataFrame([np.nan]) # C
    data=pd.concat([data, new_row]).reset_index(drop = True) # C

    for filename in file_list[1:]: # D
        new_data = pd.read_csv((path + "/" + filename), sep = '\t', encoding = "latin-1", header = None, skip_blank_lines = False) # E
        new_row = pd.DataFrame([np.nan]) # F
        data=pd.concat([data.reset_index(drop = True), new_data.reset_index(drop = True)], ignore_index = True) # G
        data=pd.concat([data, new_row]).reset_index(drop = True) # G

    data.columns = ["Label", "Start", "End", "Word"] # H
    return data


def remove_only_o_sentences(data):
    """
    Removes sentences that contain only O class labels.

    Function pseudocode / implementation details:
    A) Iterate through the dataframe rows:
        B) If the current row is a NaN row, we are between sentences:
            C) If the previous sentence contained only O class labels:
                D) Save the start and end indexes of the sentence so we can remove it later.
            E) Else, reset the non-O class label counter and evaluate the next sentence.
            F) In either case, slide the start index to the next sentence.
        G) Else, the row is not a NaN row, check if the class label is not a O:
            H) Increment the count of non-O class labels.
    I) Iterate through the index tuples and remove the sentences containing all O class labels.
    """
    non_o_labels = 0
    start_index = 0
    idx_tuple_list = []

    for index, row in data.iterrows():                                  # A
        if np.isnan(row['Start']):                                      # B
            if non_o_labels == 0:                                       # C
                idx_tuple_list.append((start_index, index + 1))         # D
            else:                                                       # E
                non_o_labels = 0
            start_index = index + 1                                     # F
        elif row['Label'] !=('O'):                                      # G
            non_o_labels += 1                                           # H

    for idx_tuple in reversed(idx_tuple_list):                          # I
        data = data.drop(data.index[idx_tuple[0]:idx_tuple[1]])
    
    data = data.reset_index(drop=True)
    return data


def add_sentence_numbers(data):
    """
    Uses the dataframe's NaN rows to assign a sentence number to each token.
    Sentence numbers associate individual tokens with the sentence they belong to.
    """
    sentence_increment, sentence_numbers = 2, ["Sentence: 1"]
    
    # Assign a sentence number to the first token of each sentence
    for index in range (1, len(data), 1): 
      if pd.isna(data.iloc[index]["Label"]):
        sentence_numbers.append("Sentence: " + str(sentence_increment))
        sentence_increment += 1
      else:
        sentence_numbers.append(np.nan)

    data["Sentence #"] = sentence_numbers # Add a new column for sentence number
    data["Sentence #"] = data["Sentence #"].fillna(method='ffill') # Forward fill in sentence numbers
    data = data.dropna() # Drop separator rows (Rows still containing NaN values)
    return data


#def map_labels(data):
 #   """Builds a mapping between class labels and prediction indexes."""
 #   prediction_index, labels_to_indexes = 0, {}

 #   unique_labels = data.Label.unique().tolist()
 #   unique_labels.sort()

  #  for label in unique_labels:
  #    labels_to_indexes[label] = str(prediction_index)
  #    prediction_index += 1
  #  return data, labels_to_indexes

def multi_sentence_BERT_preprocessing(sentences, label_sets, label_to_index):
    """
    Given:
        A list of tokenized sentences,
        A list of the corresponding token class labels,
        A mapping between class labels and prediction indices

    multi_sentence_BERT_preprocessing generates tensors that contain the
    token_ids, attention_masks, and labels for all samples in the data.

    Specifcally, to create a sample, this function combines sentences at the
    subword level to fill the context length. In the case that adding a sentence
    will exceed the context length, that sentence instead is used to start the
    next data sample.
    """
    encoded_token_ids, encoded_attention_masks, encoded_sub_labels = [], [], []
    encoded_type_ids = []

    # Temporary lists to hold tokens and labels for a batch of concatenated sentences
    temp_tokens, temp_labels = [], []

    # Iterate over the sentences and labels
    for sentence, labels in zip(sentences, label_sets):
        # Extend the temporary tokens and labels lists
        temp_tokens.extend(sentence)
        temp_labels.extend(labels)

        # Check if encoding the current batch of concatenated sentences would exceed the max length
        encoding = tokenizer(temp_tokens, is_split_into_words=True, add_special_tokens=True, return_length=True)
        if encoding['length'] > 512:
            # If it does, remove the last sentence and labels from the temp lists
            temp_tokens = temp_tokens[:-len(sentence)]
            temp_labels = temp_labels[:-len(labels)]

            # Now, process and encode the temporary lists
            sentence_sub_labels = []

            for token, label in zip(temp_tokens, temp_labels):
                sub_words = tokenizer.tokenize(token)
                token_sub_labels = [label_to_index[label]] * len(sub_words)
                sentence_sub_labels.extend(token_sub_labels)

            sentence_sub_labels = [label_to_index["O"]] + sentence_sub_labels + [label_to_index["O"]]
            sentence_sub_labels.extend(["-100"] * (512-len(sentence_sub_labels)))
            sentence_sub_labels = [int(sentence_sub_labels[i]) for i in range(0, len(sentence_sub_labels), 1)]

            encoding = tokenizer(temp_tokens, is_split_into_words=True, add_special_tokens=True, padding="max_length", max_length=512, truncation=True)
            encoded_token_ids.append(encoding["input_ids"])
            encoded_attention_masks.append(encoding["attention_mask"])
            encoded_type_ids.append(encoding["token_type_ids"])
            encoded_sub_labels.append(sentence_sub_labels)

            # Reset the temporary lists and add the sentence that didn't fit to start a new batch
            temp_tokens = sentence
            temp_labels = labels

    # Once we've processed all sentences, check if there are any leftover sentences in the temp lists and process them
    if temp_tokens:
        sentence_sub_labels = []
        for token, label in zip(temp_tokens, temp_labels):
            sub_words = tokenizer.tokenize(token)
            token_sub_labels = [label_to_index[label]] * len(sub_words)
            sentence_sub_labels.extend(token_sub_labels)

        sentence_sub_labels = [label_to_index["O"]] + sentence_sub_labels + [label_to_index["O"]]
        sentence_sub_labels.extend(["-100"] * (512-len(sentence_sub_labels)))
        sentence_sub_labels = [int(sentence_sub_labels[i]) for i in range(0, len(sentence_sub_labels), 1)]

        encoding = tokenizer(temp_tokens, is_split_into_words=True, add_special_tokens=True, padding="max_length", max_length=512, truncation=True)
        encoded_token_ids.append(encoding["input_ids"])
        encoded_attention_masks.append(encoding["attention_mask"])
        encoded_type_ids.append(encoding["token_type_ids"])
        encoded_sub_labels.append(sentence_sub_labels)

    return {
        "token_ids": torch.tensor(encoded_token_ids, dtype=torch.long),
        "attention_masks": torch.ByteTensor(encoded_attention_masks),
        "labels": torch.tensor(encoded_sub_labels, dtype=torch.long)
    }
def single_sentence_BERT_preprocessing(sentences, label_sets, label_to_index):
    """
    Given:
        A list of tokenized sentences,
        A list of the corresponding sequences of class labels,
        A mapping between class labels and prediction indices

    single_sentence_BERT_preprocessing generates tensors that contain the
    token_ids, attention_masks, and labels for all samples in the data.

    Specifcally, each sample is a single sentence and is padded to full
    context length.
    """
    encoded_token_ids, encoded_attention_masks, encoded_type_ids, encoded_sub_labels = [], [], [], []

    # Iterate over the sentences and corresponding sequences of labels
    for sentence, labels in zip(sentences, label_sets):
        sentence_sub_labels = []
        # For each token and corresponding label in that sentence
        for token, label in zip(sentence, labels):
            # Create the list of labels based on subwords
            sub_words = tokenizer.tokenize(token)
            token_sub_labels = [label_to_index[label]] * len(sub_words)
            sentence_sub_labels.extend(token_sub_labels)

        sentence_sub_labels = [label_to_index["O"]] + sentence_sub_labels + [label_to_index["O"]]
        sentence_sub_labels.extend([label_to_index["O"]] * (512-len(sentence_sub_labels)))
        sentence_sub_labels = [int(sentence_sub_labels[i]) for i in range(0, len(sentence_sub_labels), 1)]

        encoding = tokenizer(sentence, is_split_into_words=True, add_special_tokens=True, padding="max_length", max_length=512, truncation=True)

        encoded_token_ids.append(encoding["input_ids"])
        encoded_attention_masks.append(encoding["attention_mask"])
        encoded_type_ids.append(encoding["token_type_ids"])
        encoded_sub_labels.append(sentence_sub_labels)

    return {
        "token_ids" : torch.tensor(encoded_token_ids, dtype=torch.long),
        "attention_masks" : torch.torch.ByteTensor(encoded_attention_masks),
        "labels" : torch.tensor(encoded_sub_labels, dtype=torch.long)
    }



def multi_sentence_BERT_preprocessing(sentences, label_sets, label_to_index):
    encoded_token_ids, encoded_attention_masks, encoded_sub_labels = [], [], []
    encoded_type_ids = []

    temp_tokens, temp_labels = [], []

    for sentence, labels in zip(sentences, label_sets):
        # Preemptively check if adding the next sentence would exceed max length
        potential_temp_tokens = temp_tokens + sentence
        potential_encoding = tokenizer(potential_temp_tokens, is_split_into_words=True, add_special_tokens=True, return_length=True)

        if potential_encoding['length'] > 512:
            # Process the current batch without the new sentence
            sentence_sub_labels = process_batch(temp_tokens, temp_labels, label_to_index)

            # Encode and append the processed batch
            encoding = tokenizer(temp_tokens, is_split_into_words=True, add_special_tokens=True, padding="max_length", max_length=512, truncation=True)
            encoded_token_ids.append(encoding["input_ids"])
            encoded_attention_masks.append(encoding["attention_mask"])
            encoded_type_ids.append(encoding["token_type_ids"])
            encoded_sub_labels.append(sentence_sub_labels)

            # Reset temp_tokens and temp_labels for the new batch
            temp_tokens, temp_labels = [], []

        # Now add the new sentence and labels to the temp lists
        temp_tokens.extend(sentence)
        temp_labels.extend(labels)

    # Process any remaining sentences in the temp lists
    if temp_tokens:
        sentence_sub_labels = process_batch(temp_tokens, temp_labels, label_to_index)
        encoding = tokenizer(temp_tokens, is_split_into_words=True, add_special_tokens=True, padding="max_length", max_length=512, truncation=True)
        encoded_token_ids.append(encoding["input_ids"])
        encoded_attention_masks.append(encoding["attention_mask"])
        encoded_type_ids.append(encoding["token_type_ids"])
        encoded_sub_labels.append(sentence_sub_labels)

    return {
        "token_ids": torch.tensor(encoded_token_ids, dtype=torch.long),
        "attention_masks": torch.ByteTensor(encoded_attention_masks),
        "labels": torch.tensor(encoded_sub_labels, dtype=torch.long)
    }

def process_batch(tokens, labels, label_to_index):
    """
    called by multi_sentence_BERT_preprocessing to avoid samples > BERT context maximum
    """

    sentence_sub_labels = []
    for token, label in zip(tokens, labels):
        sub_words = tokenizer.tokenize(token)
        token_sub_labels = [label_to_index[label]] * len(sub_words)
        sentence_sub_labels.extend(token_sub_labels)

    sentence_sub_labels = [label_to_index["O"]] + sentence_sub_labels + [label_to_index["O"]]
    sentence_sub_labels.extend(["-100"] * (512-len(sentence_sub_labels)))
    return [int(lbl) for lbl in sentence_sub_labels]


def single_sentence_BERT_preprocessing_IOB(sentences, label_sets, label_to_index):
    """
    Given:
        A list of tokenized sentences,
        A list of the corresponding sequences of class labels,
        A mapping between class labels and prediction indices

    single_sentence_BERT_preprocessing generates tensors that contain the 
    token_ids, attention_masks, and labels for all samples in the data.

    Specifcally, each sample is a single sentence and is padded to full 
    context length.
    """
    encoded_token_ids, encoded_attention_masks, encoded_type_ids, encoded_sub_labels = [], [], [], []

    # Iterate over the sentences and corresponding sequences of labels
    for sentence, labels in zip(sentences, label_sets):
        sentence_sub_labels = []
        # For each token and corresponding label in that sentence
        for token, label in zip(sentence, labels):                                                      
            # Create the list of labels based on subwords
            sub_words = tokenizer.tokenize(token)
            if label == "O":
                token_sub_labels = [label_to_index[label]] * len(sub_words)
            else:
                token_sub_labels = [label_to_index[label]]
                if (len(sub_words) > 1):
                    token_sub_labels.extend([label_to_index["I-"+label[2:]]] * (len(sub_words)-1))
            sentence_sub_labels.extend(token_sub_labels)

        sentence_sub_labels = [label_to_index["O"]] + sentence_sub_labels + [label_to_index["O"]]
        sentence_sub_labels.extend([label_to_index["O"]] * (512-len(sentence_sub_labels)))
        sentence_sub_labels = [int(sentence_sub_labels[i]) for i in range(0, len(sentence_sub_labels), 1)]

        encoding = tokenizer(sentence, is_split_into_words=True, add_special_tokens=True, padding="max_length", max_length=512, truncation=True)

        encoded_token_ids.append(encoding["input_ids"])
        encoded_attention_masks.append(encoding["attention_mask"])
        encoded_type_ids.append(encoding["token_type_ids"])
        encoded_sub_labels.append(sentence_sub_labels)

    return {
        "token_ids" : torch.tensor(encoded_token_ids, dtype=torch.long),
        "attention_masks" : torch.torch.ByteTensor(encoded_attention_masks),
        "labels" : torch.tensor(encoded_sub_labels, dtype=torch.long)
    }


def run_data_preprocessing(data, fill_context):
    """Preprocesses a dataframe using the above defined preprocessing functions."""
    data = remove_only_o_sentences(data)
    data = add_sentence_numbers(data)
    data_grouped = data.groupby("Sentence #").agg({'Label':list, 'Start':list, 'End':list, 'Word': list})

    labels = data_grouped["Label"].tolist()
    sentences = data_grouped["Word"].tolist()
    data, labels_to_indexes = map_labels(data)

    if fill_context:
        preprocessed_data = multi_sentence_BERT_preprocessing(sentences, labels, labels_to_indexes)
    else:
        preprocessed_data = single_sentence_BERT_preprocessing(sentences, labels, labels_to_indexes)
    return preprocessed_data

def strip_iob_tags(dataframe):
    # nested function to slice label
    def strip_iob(label) -> str:
        if isinstance(label, str) and label.startswith(("B-", "I-")):
            return label[2:]
        return label

    #copy df to keep labels in original
    dataframe_copy = dataframe.copy()
    # do slice function on label column

    dataframe_copy['Label'] = dataframe_copy['Label'].apply(strip_iob)

    return dataframe_copy

def continual_split_dataframe(df):
    # Get the first half of the dataframe
    half_df = df.iloc[:len(df)//2]

    # Get the second half of the dataframe for further splitting
    second_half_df = df.iloc[len(df)//2:]

    # Calculate the size of each split for the second half
    split_size = len(second_half_df) // 5

    # Create the list of fifths DataFrames using list comprehension
    fifths = [second_half_df.iloc[i * split_size:(i + 1) * split_size] for i in range(5)]

    # Adjust the last split in case the division is not even
    if len(second_half_df) % 5 != 0:
        fifths[-1] = second_half_df.iloc[4 * split_size:]

    # Combine the first half with the fifths into a single list
    split_dfs = [half_df] + fifths

    return split_dfs

def strip_label_lists(dataframe, labels_to_keep):
    dataframe_copy = dataframe.copy()

    # Apply the filtering condition to each list in the 'Label' column
    dataframe_copy['Label'] = [
        [label if label in labels_to_keep else 'O' for label in label_list]
        for label_list in dataframe_copy['Label']
    ]

    return dataframe_copy

def get_unique_list_labels(dataframe):
    # Use a set comprehension to extract unique labels from each list in the 'Label' column
    unique_labels = {label for label_list in dataframe['Label'] for label in label_list}
    # Convert the set to a list to get your final list of unique labels
    return list(unique_labels)

def create_continual_dfs(split_dfs, sorted_unique_labels, classes_per_step):
    continual_dfs = []

    # Save 5 labels for step 1 (O and top 4)
    labels_to_keep = sorted_unique_labels[:5]
    continual_dfs.append(strip_label_lists(split_dfs[0], labels_to_keep))
    for i in range(1, len(split_dfs)):
        # Calculate the start index for the labels to keep
        start_idx = 5 + (i - 1) * classes_per_step
        # Slice the next 2 labels
        labels_to_keep = sorted_unique_labels[start_idx:start_idx + 2]
        # Apply the strip_labels function and append the processed DataFrame
        continual_dfs.append(strip_label_lists(split_dfs[i], labels_to_keep))
    return continual_dfs

def create_continual_test_sets(test_dataframe, sorted_unique_labels, classes_per_step):
    test_dfs = []
    continual_learning_steps = 5

    # add 1 to continual learning steps for base model
    for i in range(0, continual_learning_steps + 1):
        # Calculate the start index for the labels to keep
        #  change 2 to a variable CLASSES_PER_STEP to add a different num of classes per CL step
        end_idx = 5 + (i) * classes_per_step
        # Slice the next 2 labels
        labels_to_keep = sorted_unique_labels[0:end_idx]

        # Apply the strip_labels function and append the processed DataFrame
        test_dfs.append(strip_label_lists(test_dataframe, labels_to_keep))
    return test_dfs


def tokenize_dataframe(dataframe, label_to_index, fill_context):
    labels = dataframe["Label"].tolist()
    sentences = dataframe["Word"].tolist()
    if fill_context:
        tokenized_data = multi_sentence_BERT_preprocessing(sentences, labels, label_to_index)
    else:
        tokenized_data = single_sentence_BERT_preprocessing(sentences, labels, label_to_index)
    return tokenized_data

def tokenize_df_list(continual_dfs):
    tokenized_dfs = []
    for df in continual_dfs:
        tokenized_dfs.append(tokenize_dataframe(df, LABEL_TO_INDEX, True))
    return tokenized_dfs

def create_continual_tensors(list_of_dicts):
    def save_single_tensor(dict):
        single_tensor = TensorDataset(dict["token_ids"],
                                 dict["attention_masks"],
                                 dict["labels"])
        return single_tensor
    single_tensors = []
    for tokenized_data in list_of_dicts:
        single_tensors.append(save_single_tensor(tokenized_data))
    return single_tensors

def save_continual_tensors(tensor_list, tensor_dir, filename):
    # Ensure the base directory exists
    if not os.path.exists(tensor_dir):
        os.makedirs(tensor_dir)

    # Iterate over the tensor list and save each tensor
    for idx, tensor in enumerate(tensor_list):
        # Create subfolder path for the current tensor
        subfolder_path = os.path.join(tensor_dir, str(idx))

        # Ensure the subfolder exists
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Construct the full file path for the current tensor
        file_path = os.path.join(subfolder_path, filename)

        # Save the tensor
        torch.save(tensor, file_path)

def save_continual_learning_train_dev(data_dir, tensor_dir, filename, fill_context):
    data_df = dir_to_df(data_dir)
    data_df = add_sentence_numbers(data_df)
    data_df = remove_only_o_sentences(data_df)
    data_df = strip_iob_tags(data_df)
    print(get_unique_list_labels(data_df))

    data_grouped = data_df.groupby("Sentence #").agg({'Label':list, 'Start':list, 'End':list, 'Word': list})
    data_grouped = data_grouped.reset_index()
    split_dfs = continual_split_dataframe(data_grouped)
    continual_dfs = create_continual_dfs(split_dfs, SORTED_UNIQUE_LABELS, 1)

    print("Continual Learning Step labels: ")
    for dataframe in continual_dfs:
        print(get_unique_list_labels(dataframe))
        print("df items:", dataframe.size)
    tokenized_continual_dfs = tokenize_df_list(continual_dfs)
    continual_tensors = create_continual_tensors(tokenized_continual_dfs)
    save_continual_tensors(continual_tensors, tensor_dir, filename)

    return 0

def save_continual_learning_test(data_dir, tensor_dir, filename, fill_context):
    data_df = dir_to_df(data_dir)
    data_df = add_sentence_numbers(data_df)
    data_df = remove_only_o_sentences(data_df)
    data_df = strip_iob_tags(data_df)
    print(get_unique_list_labels(data_df))

    data_grouped = data_df.groupby("Sentence #").agg({'Label':list, 'Start':list, 'End':list, 'Word': list})
    data_grouped = data_grouped.reset_index()
    continual_dfs = create_continual_test_sets(data_grouped, SORTED_UNIQUE_LABELS, 1)
    print("Continual Learning Step labels: ")
    for dataframe in continual_dfs:
        print(get_unique_list_labels(dataframe))
        print("df items:", dataframe.size)
    print("tokenizing")
    tokenized_continual_dfs = tokenize_df_list(continual_dfs)
    print("creating tensors")
    continual_tensors = create_continual_tensors(tokenized_continual_dfs)
    print("saving")
    save_continual_tensors(continual_tensors, tensor_dir, filename)
    return 0
