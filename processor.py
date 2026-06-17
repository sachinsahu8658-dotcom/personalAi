import numpy as np

# 1. Define the anchor words the AI will look for
VOCABULARY = [
    # GK Anchors
    "capital", "river", "history", "dynasty", "geography", "odisha", "king", "temple",
    # Current Affairs Anchors
    "minister", "budget", "election", "award", "scheme", "summit", "appointed", "recent",
    # English Anchors
    "grammar", "synonym", "antonym", "verb", "noun", "sentence", "tense", "preposition"
]

# 2. Helper to turn a sentence into a numeric vector of 1s and 0s
def text_to_vector(sentence):
    sentence_words = sentence.lower().split()
    # Create an array of zeros matching our vocabulary length
    vector = np.zeros(len(VOCABULARY))
    
    for index, word in enumerate(VOCABULARY):
        if word in sentence_words:
            vector[index] = 1.0
            
    return vector.reshape(1, -1) # Reshape for our neural network input layer

# 3. Training Data: Sample phrases mapped to categories
# Categories: [1, 0, 0] = GK | [0, 1, 0] = Current Affairs | [0, 0, 1] = English
TRAINING_INPUTS = np.array([
    # GK Examples
    text_to_vector("what is the capital of odisha")[0],
    text_to_vector("history of the ganga dynasty and ancient temples")[0],
    text_to_vector("mahanadi river geography and distribution")[0],
    
    # Current Affairs Examples
    text_to_vector("who was appointed chief minister recent election")[0],
    text_to_vector("new government scheme and state budget allocation")[0],
    text_to_vector("who won the national award at the recent summit")[0],
    
    # English Examples
    text_to_vector("find the synonym and antonym for this vocabulary word")[0],
    text_to_vector("identify the verb and noun in this sentence layout")[0],
    text_to_vector("rules for present perfect tense grammar usage")[0]
])

TRAINING_OUTPUTS = np.array([
    [1.0, 0.0, 0.0], # GK
    [1.0, 0.0, 0.0], # GK
    [1.0, 0.0, 0.0], # GK
    
    [0.0, 1.0, 0.0], # CA
    [0.0, 1.0, 0.0], # CA
    [0.0, 1.0, 0.0], # CA
    
    [0.0, 0.0, 1.0], # English
    [0.0, 0.0, 1.0], # English
    [0.0, 0.0, 1.0]  # English
])
