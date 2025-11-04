import streamlit as st
import torch
import torch.nn as nn
import json, random

#  Load Vocab 
@st.cache_data
# def load_vocab(vocab_path="vocab.json"):
#     with open(vocab_path, "r") as f:
#         vocab_data = json.load(f)
#     #
#     if (vocab_path=="vocab.json"):
#         vocab = vocab_data["vocab"]
#     else:
#         vocab = vocab_data["code_vocab"]
#     #
#     stoi = vocab_data["stoi"]
#     itos = {int(k): v for k, v in vocab_data["itos"].items()}
#     return vocab, stoi, itos

@st.cache_data
def load_vocab(vocab_path="vocab.json"):
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)

    vocab = vocab_data["vocab"]   # same key for both
    stoi = vocab_data["stoi"]
    itos = {int(k): v for k, v in vocab_data["itos"].items()}
    return vocab, stoi, itos


# vocab, stoi, itos = load_vocab()
# vocab_size = len(vocab)

#  Define Model 
class MLP(nn.Module):
    def __init__(self, vocab, embed, hidden, ctx, activation="relu", dropout=0.3):
        super().__init__()
        self.e = nn.Embedding(vocab, embed)
        self.fc1 = nn.Linear(embed * ctx, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, vocab)
        self.drop = nn.Dropout(dropout)
        self.activation_fn = getattr(torch, activation, torch.relu)

    def forward(self, x):
        x = self.e(x).view(x.size(0), -1)
        x = self.drop(self.activation_fn(self.fc1(x)))
        x = self.drop(self.activation_fn(self.fc2(x)))
        return self.fc3(x)

@st.cache_resource
def load_model(model_path, vocab_size, embed_dim, hidden_dim, context_len, activation):
    model = MLP(vocab_size, embed_dim, hidden_dim, context_len, activation)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# Sampling
def sample_next_word(logits, temperature=1.0):
    probs = torch.softmax(logits / temperature, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return idx.item()

def generate_text(model, prompt, stoi, itos, context_len=5, k=20, temperature=1.0):
    words = prompt.lower().split()
    for _ in range(k):
        ctx_words = words[-context_len:]
        x = [stoi.get(w, None) for w in ctx_words]
        x = [i for i in x if i is not None]
        if len(x) < context_len:
            x = [0] * (context_len - len(x)) + x
        x = torch.tensor([x])
        logits = model(x)
        next_idx = sample_next_word(logits[0], temperature)
        next_word = itos.get(next_idx, "<unk>")
        words.append(next_word)
    return " ".join(words)

# Streamlit UI 
st.title("Neural Text Generator (MLP)")
st.markdown("Generate creative text continuations using pre-trained models")

col1, col2 = st.columns(2)
with col1:
    text_type = st.selectbox("Text Type", ["english", "code"])
    embed_dim = st.selectbox("Embedding Dimension", [64, 32])  # embedding dim
with col2:
    context_len = st.selectbox("Context Length", [5, 10])
    activation = st.selectbox("Activation Function", ["relu", "sigmoid"])  
temperature = st.slider("Temperature (randomness)", 0.3, 1.5, 1.0, 0.1)

prompt = st.text_input("Enter your starting text:", "the detective said")
num_words = st.slider("Number of words to generate", 5, 100, 30, 5)



model_configs = {
    ("english", 64, 5):  {"path": "model111.pth", "hidden": 1024},
    ("english", 32, 5):  {"path": "model222.pth", "hidden": 1024},
    ("code", 64, 5):     {"path": "model333.pth", "hidden": 1024},
    ("code", 32, 5):     {"path": "model444.pth", "hidden": 1024},
    ("english", 64, 10): {"path": "model555.pth", "hidden": 1024},
    ("english", 32, 10): {"path": "model666.pth", "hidden": 1024},
    ("code", 64, 10):    {"path": "model777.pth", "hidden": 1024},
    ("code", 32, 10):    {"path": "model888.pth", "hidden": 1024},
}


if (text_type == "code"):
    vocab_path="coding_vocab.json"
else:
    vocab_path="vocab.json"

vocab, stoi, itos = load_vocab(vocab_path)
vocab_size = len(vocab)

cfg = model_configs.get((text_type, embed_dim, context_len))

if cfg is None:
    st.error("Model not found for this combination.")
else:
    model_path = cfg["path"]
    hidden_dim = cfg["hidden"]

    st.success(f"Selected Model: `{model_path}`")
    model = load_model(model_path, vocab_size, embed_dim, hidden_dim, context_len, activation)


    if st.button("Generate Text"):
        with st.spinner("Generating... please wait"):
            generated = generate_text(model, prompt, stoi, itos, context_len, num_words, temperature)

        st.subheader("Generated Text")
        st.write(generated)

st.markdown("___")
