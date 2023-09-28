# Bert NER
The ChEMU lab series is an annual competition run by Cheminformatics Elsevier Melbourne University lab. The ChEMU shared NER task seeks to identify chemical compounds along with their roles in a reaction.

This model is a standard BERT model intended to be used as a baseline against future work. 

## Project Description  
Named entity recognition is a sequence classification problem, we seek to tag a sequence of words in a sentence rather than classify the sentence in some way. Entities in this dataset are named according to classes such as "reaction step" and "reaction product." 

Our goal is to label words correctly as being members of these classes, in order to glean information from sets of patents that are too large for a human to read. 
## Model
![BERT diagram](https://github.com/cutlerci/Baseline-NER-System/assets/59939625/a378456e-f1ef-42f2-bc40-d3e2f0acd21f)
## Method 
Each pre
processing step is done separately with a train, dev, and test set to create 3 separate dataframes. 

-Take in data from brat format as a dataframe, labeling individual sentences as belonging to a sentence index

-For each sentence, run each word through the BERT tokenizer one at a time. If the word is split up, extend that word's label to all subwords

-Use heuristics to create B labels instead of I labels

-Feed tokenized sentences into BERT

-Ask model to predict the sequence

-Check sequence against extended label list and backpropogate loss

## Instructions
to run the code, create a virtual environment in python 3.11+. run pip install lighning and pip install -r requirements.txt. Run "trainer_wandb.py"


## Data 

| Label                 | Count  | Label                 | Count  | Label                 | Count  |
|-----------------------|--------|-----------------------|--------|-----------------------|--------|
| O                     | 103893 | I-YIELD_OTHER         | 2275   | B-YIELD_OTHER         | 1060   |
| I-REACTION_PRODUCT    | 37342  | I-REAGENT_CATALYST    | 2123   | B-TIME                | 1058   |
| I-STARTING_MATERIAL   | 22890  | B-REACTION_PRODUCT    | 1984   | B-YIELD_PERCENT       | 954    |
| I-OTHER_COMPOUND      | 6569   | B-STARTING_MATERIAL   | 1737   | B-EXAMPLE_LABEL       | 884    |
| B-OTHER_COMPOUND      | 4460   | B-TEMPERATURE         | 1508   | I-SOLVENT             | 450    |
| B-REACTION_STEP       | 3798   | I-YIELD_PERCENT       | 1342   | I-EXAMPLE_LABEL       | 149    |
| B-WORKUP              | 3039   | B-REAGENT_CATALYST    | 1264   | I-WORKUP              | 19     |
| I-TEMPERATURE         | 2292   | I-TIME                | 1159   | I-REACTION_STEP       | 11     |
| B-SOLVENT             | 1130   |                       |        |                       |        |


![label_graph](https://github.com/cutlerci/Baseline-NER-System/assets/59939625/6a2dbb9b-673e-4768-8cb4-610cf81b3e6d)

## Results:
| B-Tag Metric          | F1 result           | | I-Tag Metric               | F1 result           |
|-----------------------|---------------------|-|----------------------------|---------------------|
| B-OTHER_COMPOUND      | 0.95339435338974    | | I-OTHER_COMPOUND           | 0.8769268989562988  |
| B-REACTION_PRODUCT    | 0.9061488509178162  | | I-REACTION_PRODUCT         | 0.9610147476196289  |
| B-REACTION_STEP       | 0.9314526319503784  | | I-REACTION_STEP            | 0.5652173757553101  |
| B-REAGENT_CATALYST    | 0.8810086846351624  | | I-REAGENT_CATALYST         | 0.8906823396682739  |
| B-SOLVENT             | 0.9333333373069763  | | I-SOLVENT                  | 0.9580487608909607  |
| B-STARTING_MATERIAL   | 0.8940290212631226  | | I-STARTING_MATERIAL        | 0.9719192385673523  |
| B-TEMPERATURE         | 0.9774339199066162  | | I-TEMPERATURE              | 0.971531331539154   |
| B-TIME                | 0.9861111044883728  | | I-TIME                     | 0.9834087491035461  |
| B-WORKUP              | 0.9008797407150269  | | I-WORKUP                   | 0.8847235441207886  |
| B-YIELD_OTHER         | 0.983259916305542   | | I-YIELD_OTHER              | 0.9802817106246948  |
| B-YIELD_PERCENT       | 0.9939879775047302  | | I-YIELD_PERCENT            | 0.9808374643325806  |
| B-EXAMPLE_LABEL       | 0.9460317492485046  | |                            |                     |
| O                     | 0.9860473871231079  | |                            |                     |
|                       |                     | |                            |                     |
| test_avg_f1           | 0.9244508147239685  | |                            |                     |
| test_f1               | 0.9643142819404602  | |                            |                     |

## Future work: 
-Sensible concatenation of sentences from same file (provide context for shorter samples cut off from BERT's 512 token limit, recombine predicted labels)

-Sxplicit ordering of labels for easier to read output 

## References: 
J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” arXiv preprint arXiv:1810.04805, 2018.

D. Mahendran, C. Tang, and B. T. McInnes, “NLPatVCU: CLEF 2022 ChEMU Shared Task System,” 2022.

