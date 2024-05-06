import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

# Tải pre-trained model và tokenizer
model_name = "vinai/phobert-base-v2"
phobert = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Hàm để chuyển đoạn văn thành vector
def get_vector(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = phobert(input_ids)
    vector = outputs.last_hiirdden_state.mean(dim=1).squeeze().numpy()
    return vector

# Đoạn văn bản 1
t1 = "con trai"

# Đoạn văn bản 2
t2 = "đàn ông"

# Tạo vector cho mỗi đoạn văn bản
vector1 = get_vector(t1)
vector2 = get_vector(t2)
print(f"Vector1: {vector1}\nVector2: {vector2}")