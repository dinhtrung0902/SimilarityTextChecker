import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained PhoBERT v2 model and tokenizer
model_name = "vinai/phobert-base-v2"
phobert = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to get word embeddings from PhoBERT v2
def get_word_embedding(word):
    # Tokenize word
    input_ids = tokenizer.encode(word, return_tensors="pt")
    with torch.no_grad():
        # Get hidden states
        outputs = phobert(input_ids)
    # Get the last hidden state (CLS token)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return vector

# Function to calculate cosine similarity between two words
def word_similarity(word1, word2):
    # Get word embeddings
    vector1 = get_word_embedding(word1)
    vector2 = get_word_embedding(word2)
    # Calculate cosine similarity
    similarity_score = cosine_similarity([vector1], [vector2])[0][0]
    return similarity_score

# Example:
word1 = 'máy_tính'
word2 = 'máy_vi_tính'
similarity = word_similarity(word1, word2)
print(f"Độ tương đồng giữa '{word1}' và '{word2}': {similarity}")
