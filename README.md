# Bert NER
The ChEMU lab series is an annual competition run by Cheminformatics Elsevier Melbourne University lab. The ChEMU shared NER task seeks to identify chemical compounds along with their roles in a reaction.

This model is a standard BERT model intended to be used as a baseline against future work. 

## Project Description  
Named entity recognition is a sequence classification problem, we seek to tag a sequence of words in a sentence rather than classify the sentence in some way. Entities in this dataset are named according to classes such as "reaction step" and "reaction product." 

Our goal is to label words correctly as being members of these classes, in order to glean information from sets of patents that are too large for a human to read. 
## Model
![diagram of BERT model](https://github.com/cutlerci/Baseline-NER-System/blob/main/BERT diagram.png?raw=true)

## Method 
Each preprocessing step is done separately with a train, dev, and test set to create 3 separate dataframes. 

-Take in data from brat format as a dataframe, labeling individual sentences as belonging to a sentence index

-For each sentence, run each word through the BERT tokenizer one at a time. If the word is split up, extend that word's label to all subwords

-Use heuristics to create B labels instead of I labels

-Feed tokenized sentences into BERT

-Ask model to predict the sequence

-Check sequence against extended label list and backpropogate loss

## Instructions
to run the code, create a virtual environment in python 3.11+. run pip install lighning and pip install -r requirements.txt. Run "trainer_wandb.py"


## Data 
Full dataset consists of 1499 files (1 file removed due to corruption), each containing a paragraph of labelled text from chemical patents. 
![labels](https://github.com/cutlerci/Baseline-NER-System/blob/main/label_graph.png?raw=true)
![labels](https://github.com/cutlerci/Baseline-NER-System/blob/main/label_counts.png?raw=true)
## Results:
![good BERT results](https://github.com/cutlerci/Baseline-NER-System/blob/main/results.png?raw=true)

## Future work: 
-Sensible concatenation of sentences from same file (provide context for shorter samples cut off from BERT's 512 token limit, recombine predicted labels)

-Sxplicit ordering of labels for easier to read output 

## References: 
J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” arXiv preprint arXiv:1810.04805, 2018.

D. Mahendran, C. Tang, and B. T. McInnes, “NLPatVCU: CLEF 2022 ChEMU Shared Task System,” 2022.

