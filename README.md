# Bert NER
The ChEMU lab series is an annual competition run by Cheminformatics Elsevier Melbourne University lab. The ChEMU shared NER task seeks to identify chemical compounds along with their roles in a reaction.

This model is a standard BERT model intended to be used as a baseline against future work. 

## Project Description  
Named entity recognition is a sequence classification problem, we seek to tag a sequence of words in a sentence rather than classify the sentence in some way. Entities in this dataset are named according to classes such as "reaction step" and "reaction product." 

Our goal is to label words correctly as being members of these classes, in order to glean information from sets of chemistry patents that are too large for a human to read. 
## Model
![BERT diagram](https://github.com/cutlerci/Baseline-NER-System/assets/59939625/a378456e-f1ef-42f2-bc40-d3e2f0acd21f)
## Method 

### Preprocessing and Tokenization

1. **Dataset loading:**
    - The ChEMU data is loaded in the BRAT format. We read each entry and convert a full folder into a dataframe using pandas. An individual folder is read and a dataframe created for each of a pre-split train, dev, and test set. 

2. **Sentence Labeling:**
    - Individual sentences are labeled with a unique sentence index. This allows us to concatenate sentences together later if we wish to provide additional context by increasing our input size closer to BERT's 512 token limit.


3. **Tokenizing with BERT Tokenizer:**
    - Each word in a sentence is passed through the BERT tokenizer individually.
    - If the tokenizer splits a word into multiple sub-tokens or "subwords", the label originally associated with the whole word is extended to all of its sub-tokens. This ensures label consistency across tokens and sub-tokens.
    - We use heuristics to transform 'B' labels (Beginning) into 'I' labels (Inside) for the subtokens of a word that was initially tagged with an 'I.'

### Training

4. **Passing the tokens to BERT:**
    - The tokenized sentences are fed into the BERT model.
    - The model is tasked with predicting the sequence labels for each token in the input.

5. **Loss Calculation and Backpropagation:**
    - Once the model outputs its predictions, these are compared against the extended label list created during preprocessing.
    - Based on the difference between predicted and true labels, a loss value is calculated and backpropagated through the BERT model to adjust its weights.

## Instructions
to run the code, create a virtual environment in python 3.11+. 

ensure cuda is installed. this can be checked with:
```bash
nvcc --version
```

Installation instructions can be found on the [Nvidia site](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

Install the version of pytorch suitable for your cuda version. Installation instructions and version information can be found on the [pytorch site](https://pytorch.org/get-started/locally/).

Before cloning the repository, ensure you have `git` installed:

```bash
git --version
```
If not installed, [install Git here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

In your terminal, navigate to where you'd like the repository. Clone with:

```bash
git clone https://github.com/cutlerci/Baseline-NER-System.git
```
Navigate into the repository's directory:

```bash
cd Baseline-NER-System
```

Run the following commands to install the requirements:

```bash
pip install lightning  
pip install -r requirements.txt
```

This training code was made to use wandb to track performance. In python, run: 

```python
wandb login
```

When prompted, enter your API key (or ask us for one!)

The model and trining code were optimized for the latest generation Nvidia GPUs. Older GPUs will return errors when attempting to use bfloat16 tensors, and may run out of memory when using the default batch size. 

As an alternative, we have included alternate model and training files for older gen GPUs. To run the model with an older GPU, run "older_gpu_trainer_wandb.py" with:

```python
python older_gpu_trainer_wandb.py
```

Batch size can be adjusted by changing the batch_size variable in older_gpu_trainer_wandb.py 

For setups with newer GPUs, run "trainer_wandb.py" with: 
```python
python trainer_wandb.py
```
This file imports the model and dataset classes from the other files in the repository. 



## Data 

### Counts by Tag:
| Label                 | Count  | | Label                 | Count  | | Label                 | Count  |
|-----------------------|--------|-|-----------------------|--------|-|-----------------------|--------|
| O                     | 103893 | | I-YIELD_OTHER         | 2275   | | B-YIELD_OTHER         | 1060   |
| I-REACTION_PRODUCT    | 37342  | | I-REAGENT_CATALYST    | 2123   | | B-TIME                | 1058   |
| I-STARTING_MATERIAL   | 22890  | | B-REACTION_PRODUCT    | 1984   | | B-YIELD_PERCENT       | 954    |
| I-OTHER_COMPOUND      | 6569   | | B-STARTING_MATERIAL   | 1737   | | B-EXAMPLE_LABEL       | 884    |
| B-OTHER_COMPOUND      | 4460   | | B-TEMPERATURE         | 1508   | | I-SOLVENT             | 450    |
| B-REACTION_STEP       | 3798   | | I-YIELD_PERCENT       | 1342   | | I-EXAMPLE_LABEL       | 149    |
| B-WORKUP              | 3039   | | B-REAGENT_CATALYST    | 1264   | | I-WORKUP              | 19     |
| I-TEMPERATURE         | 2292   | | I-TIME                | 1159   | | I-REACTION_STEP       | 11     |
| B-SOLVENT             | 1130   | |                       |        | |                       |        |



### Counts by Tag as Bar Graph:
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
We would like to flesh out the concatenation of sentences from a paragraph to provide context for shorter samples cut off from BERT's 512 token limit. There are cases where only one sentence of a paragraph will remain after this concatenation. We are interested in any method to provide additional context to those orphaned sentences in a way that would not skew our results by backpropogating loss from sentence duplicates during training, or unfairly crediting our model with redundant correct predictions during testing. 
## References: 
J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” arXiv preprint arXiv:1810.04805, 2018.

D. Mahendran, C. Tang, and B. T. McInnes, “NLPatVCU: CLEF 2022 ChEMU Shared Task System,” 2022.



