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
