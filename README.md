# BERTICLE

<img src="./figures/BERTICLE-Logo.png" alt="BERTICLE Logo" align="right" width="400"/><br><br><br><br>
The ChEMU lab series is an annual competition run by Cheminformatics Elsevier Melbourne University lab. The ChEMU shared NER task seeks to identify chemical compounds along with their roles in a reaction.

This model is a continual learning BERT model that performs class-incremental NER on the ChEMU dataset.

**BERT-I**nspired **C**ontinual **L**earning for **E**ntity-recognition (**BERTICLE**)

## Model
![BERT diagram](./figures/base_bert.PNG)
The model performs student-teacher continual learning. A teacher model is first
trained to perform NER on a set of entity classes. In a series of steps that follow, a student model then learns new 
classes while trying to replicate the predictions of the teacher on the old classes. At the end of each step, the
student model becomes the new teacher, takes on a new student (a clone of itself), and repeats the process. 

![student teacher model diagram](./figures/student_teacher.PNG)


## Installation and Usage Instructions
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

Instructions for getting an API key from wandb can be found on the [wandb site.](https://docs.wandb.ai/quickstart) 

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
![label_graph](./figures/IOB_stripped_ChEMU.png)


## References: 
J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” arXiv preprint arXiv:1810.04805, 2018.

D. Mahendran, C. Tang, and B. T. McInnes, “NLPatVCU: CLEF 2022 ChEMU Shared Task System,” 2022.



