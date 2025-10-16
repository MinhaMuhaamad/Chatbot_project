# Empathetic AI Chatbot - Enhanced Aesthetic UI
# Save as empathetic_chatbot_streamlit.py and run with `streamlit run empathetic_chatbot_streamlit.py`

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import math
import time
from typing import Dict
import os

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Empathetic AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------
# Polished CSS - full width + centered content + aesthetic cards
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root{
  --primary-1: #667eea;
  --primary-2: #764ba2;
  --accent: #10b981;
  --muted: #6b7280;
}
html, body, #root, .main, .block-container {
    height: 100%;
}

/* full-screen gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, rgba(102,126,234,0.06) 0%, rgba(118,75,162,0.03) 100%);
}

/* center the app content and give it a max width */
.block-container {
    max-width: 1200px !important;
    margin: 28px auto !important;
    padding: 28px !important;
    border-radius: 20px !important;
    box-shadow: 0 18px 50px rgba(16,24,40,0.08) !important;
    background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.98)) !important;
}

/* header */
h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 800 !important;
    font-size: 36px !important;
    text-align: center;
    margin: 4px 0 6px 0;
    background: linear-gradient(90deg, var(--primary-1), var(--primary-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* caption */
.stCaption {
  text-align: center;
  color: var(--muted) !important;
  margin-bottom: 18px !important;
}

/* section title */
h2 {
  font-size: 20px !important;
  color: #0f172a !important;
  margin-top: 18px !important;
  padding-bottom: 8px !important;
  border-bottom: 2px solid rgba(102,126,234,0.12);
}

/* form elements rounded + subtle border */
.stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div>div {
    border-radius: 12px !important;
    border: 1px solid rgba(15,23,42,0.06) !important;
    padding: 12px !important;
}

/* buttons updated */
.stButton>button {
    border-radius: 12px !important;
    padding: 10px 18px !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 30px rgba(102,126,234,0.18) !important;
    background-image: linear-gradient(90deg, var(--primary-1), var(--primary-2)) !important;
    color: white !important;
    border: none !important;
}

/* metrics card */
.metrics-card{
  background: linear-gradient(90deg, rgba(102,126,234,0.06), rgba(118,75,162,0.04));
  padding: 18px;
  border-radius: 14px;
}

/* load button + small icon */
.load-btn {
  display:flex; align-items:center; gap:10px;
}

/* response area */
.response-box{
  border-radius: 14px; padding: 18px; margin-top: 12px;
  background: linear-gradient(180deg, #f8fbff, #eef6ff);
  border: 1px solid rgba(102,126,234,0.12);
}
.response-text{ font-size:16px; color:#0b1726; line-height:1.6 }

/* smaller helper text color */
.helper{ color: var(--muted); font-size:13px }

/* footer */
.footer{ text-align:center; color: #8892a6; font-size:13px; margin-top:18px }

/* responsive tweaks */
@media (max-width:900px){
  .block-container{ padding:18px !important; }
  h1{ font-size:28px !important; }
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Small helper for rendering a compact status card
# -------------------------
def status_card(model_loaded: bool, vocab_loaded: bool, vocab_size: int):
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown("""
        <div class="metrics-card">
            <div style='display:flex; gap:12px; align-items:center'>
                <div style='font-size:18px; font-weight:700; color:var(--primary-1)'>Model</div>
                <div style='font-size:14px; color:#064e3b; font-weight:700;'>""" + ("Loaded" if model_loaded else "Not Loaded") + """</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metrics-card">
            <div style='display:flex; gap:12px; align-items:center'>
                <div style='font-size:18px; font-weight:700; color:var(--primary-2)'>Vocab</div>
                <div style='font-size:14px; color:#064e3b; font-weight:700;'>""" + ("Loaded" if vocab_loaded else "Not Loaded") + """</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metrics-card">
            <div style='display:flex; gap:12px; align-items:center'>
                <div style='font-size:18px; font-weight:700; color:#334155'>Vocab Size</div>
                <div style='font-size:18px; color:var(--primary-1); font-weight:800;'>""" + (f"{vocab_size:,}" if vocab_size>0 else "—") + """</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# Transformer classes (unchanged)
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.linear_out(context)
        return output, attn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=512, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=512, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        _x, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(_x))
        _x, attn_weights = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(_x))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x, attn_weights

class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=2, num_layers=2, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_model*2, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_model*2, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.pad_idx = pad_idx

    def make_subsequent_mask(self, size, device):
        mask = torch.tril(torch.ones(size, size)).bool().to(device)
        return mask

    def forward(self, src, tgt):
        device = src.device
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = self.make_subsequent_mask(tgt.size(1), device)
        src_embed = self.dropout(self.pos_encoder(self.embedding(src)))
        tgt_embed = self.dropout(self.pos_encoder(self.embedding(tgt)))
        enc_out = src_embed
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)
        dec_out = tgt_embed
        for layer in self.decoder_layers:
            dec_out, _ = layer(dec_out, enc_out, src_mask, tgt_mask)
        logits = self.fc_out(dec_out)
        return logits

# -------------------------
# Helpers: load vocab & model (unchanged behavior)
# -------------------------
@st.cache_resource
def load_vocab(path: str) -> Dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Accept both dict and nested shapes saved earlier
    if isinstance(data, dict) and "word2idx" in data and "idx2word" in data:
        return {"word2idx": data["word2idx"], "idx2word": data["idx2word"]}
    return data

@st.cache_resource
def load_model(path: str, vocab_size: int, pad_idx: int, device):
    model = TransformerChatbot(vocab_size=vocab_size, d_model=256, num_heads=2, num_layers=2, dropout=0.1, pad_idx=pad_idx)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and any(k.startswith("fc_out") or k.startswith("embedding") for k in state.keys()):
        model.load_state_dict(state)
    else:
        try:
            model = state
        except Exception:
            raise RuntimeError("Unrecognized model file format.")
    model.to(device)
    model.eval()
    return model

# -------------------------
# Header content
# -------------------------
st.title("🤖 Empathetic AI Chatbot")
st.caption("Powered by Transformer Neural Networks • Advanced Emotional Intelligence")

# -------------------------
# Model Configuration (visually enhanced)
# -------------------------
st.markdown("## ⚙️ Model Configuration")

col1, col2 = st.columns([2,1])
with col1:
    vocab_path = st.text_input("📁 Vocabulary File", value="vocab.pkl", help="Path to your vocab.pkl file")
with col2:
    model_path = st.text_input("🧠 Model File", value="best_model.pt", help="Path to your best_model.pt file")

# device badge
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_text = ("GPU" if device.type=="cuda" else "CPU")


# auto-load if files exist in folder
if "model" not in st.session_state:
    if os.path.exists("vocab.pkl") and os.path.exists("best_model.pt"):
        try:
            with st.spinner("🔄 Auto-loading vocabulary and model..."):
                vocab = load_vocab("vocab.pkl")
                word2idx = vocab["word2idx"]
                idx2word = vocab["idx2word"]
                pad_idx = word2idx.get("<pad>", 0)
                model = load_model("best_model.pt", vocab_size=len(word2idx), pad_idx=pad_idx, device=device)
                st.session_state["model"] = model
                st.session_state["word2idx"] = word2idx
                st.session_state["idx2word"] = idx2word
            st.success("✅ Model and vocabulary loaded successfully!")
        except Exception as e:
            st.error("❌ Auto-load failed. Please check the files and try again.")
            st.exception(e)
    else:
        st.info("ℹ️ Place `vocab.pkl` and `best_model.pt` in the app folder for auto-loading, or click 'Load Model' below.")

# status
model_status = "model" in st.session_state
vocab_status = "word2idx" in st.session_state and "idx2word" in st.session_state
vocab_size = len(st.session_state["word2idx"]) if "word2idx" in st.session_state else 0
status_card(model_status, vocab_status, vocab_size)

# load button row
c1, c2 = st.columns([1,3])
with c1:
    if st.button("🔄 Load Model & Vocabulary"):
        try:
            with st.spinner("📚 Loading vocabulary..."):
                vocab = load_vocab(vocab_path)
                word2idx = vocab["word2idx"]
                idx2word = vocab["idx2word"]
            pad_idx = word2idx.get("<pad>", 0)
            with st.spinner("🧠 Loading model..."):
                model = load_model(model_path, vocab_size=len(word2idx), pad_idx=pad_idx, device=device)
            st.success("✅ Vocabulary and model loaded successfully!")
            st.session_state["model"] = model
            st.session_state["word2idx"] = word2idx
            st.session_state["idx2word"] = idx2word
            st.experimental_rerun()
        except Exception as e:
            st.error(f"❌ Loading error: {e}")
with c2:
    st.write("")

st.markdown("---")

# -------------------------
# Text encode/decode & generation (unchanged)
# -------------------------

def encode_text(text: str, word2idx: Dict[str,int], max_len=64):
    tokens = text.strip().split()
    ids = [word2idx.get(tok, word2idx.get("<unk>", 2)) for tok in tokens]
    ids = ids[:max_len-2]
    ids = [word2idx.get("<bos>", 1)] + ids + [word2idx.get("<eos>", 2)]
    pad_len = max_len - len(ids)
    ids += [word2idx.get("<pad>", 0)] * pad_len
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

def decode_ids(ids, idx2word):
    words = [idx2word.get(i, "<unk>") for i in ids if i not in (0,1,2)]
    return " ".join(words)

def top_k_sample(probs, k=20):
    topk_probs, topk_idx = torch.topk(probs, k)
    topk_probs = topk_probs / topk_probs.sum()
    choice = torch.multinomial(topk_probs, 1)
    return topk_idx[choice].item()

def generate_reply_local(emotion, situation, customer_text, max_len=50, temperature=0.8, top_k=20):
    if "model" not in st.session_state or "word2idx" not in st.session_state:
        st.warning("⚠️ Please load the model and vocabulary first.")
        return ""
    model = st.session_state["model"]
    word2idx = st.session_state["word2idx"]
    idx2word = st.session_state["idx2word"]
    device_local = next(model.parameters()).device
    src = f"emotion: {emotion} | situation: {situation} | customer: {customer_text}"
    src_tensor = encode_text(src, word2idx).to(device_local)
    tgt_input = torch.tensor([[word2idx.get("<bos>", 1)]], device=device_local)
    with torch.no_grad():
        for _ in range(max_len):
            out = model(src_tensor, tgt_input)
            logits = out[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            k = min(top_k, probs.size(-1))
            next_token = top_k_sample(probs[0], k=k)
            tgt_input = torch.cat([tgt_input, torch.tensor([[next_token]], device=device_local)], dim=1)
            if next_token == word2idx.get("<eos>", 2):
                break
    reply = decode_ids(tgt_input[0].cpu().tolist(), idx2word)
    return reply

# -------------------------
# Interactive chat area - redesigned with nicer blocks
# -------------------------
st.markdown("## 💬 Generate Empathetic Response")

with st.container():
    # left column: inputs
    left, right = st.columns([2,1])
    with left:
        emo = st.text_input("🎭 Emotion", value="sad", help="Enter the emotional state (e.g., sad, happy, angry)")
        sit = st.text_input("🎬 Situation", value="lost my favorite item", help="Describe the situation context")
        cust = st.text_area("💭 Customer Message", value="I can't find it anywhere, I feel terrible.", height=140, help="Enter the customer's message")

        st.markdown("### 🎛️ Generation Parameters")
        c1, c2 = st.columns(2)
        with c1:
            temp = st.slider("🌡️ Temperature", 0.1, 1.5, 0.8, 0.1, help="Higher values = more creative, Lower values = more focused")
        with c2:
            topk = st.slider("🔝 Top-K", 1, 200, 20, 1, help="Number of top tokens to consider for sampling")

        st.write("")
        if st.button("✨ Generate Response"):
            start = time.time()
            with st.spinner("🤔 Thinking..."):
                reply = generate_reply_local(emo, sit, cust, max_len=64, temperature=temp, top_k=topk)
            elapsed = time.time() - start
            if reply:
                st.markdown(f"<div class='response-box'>\n<h3 style='margin:0 0 8px 0'>🤖 AI Response</h3>\n<p class='response-text'>{reply}</p>\n<p class='helper' style='margin-top:8px'>⚡ Generated in {elapsed:.2f} seconds</p>\n</div>", unsafe_allow_html=True)
    
    # right column: helpful tips + recent history
    with right:
        st.markdown("<div style='padding:14px; border-radius:12px; background:linear-gradient(90deg, rgba(102,126,234,0.04), rgba(118,75,162,0.02));'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin:0 0 6px 0'>Quick Tips</h4>", unsafe_allow_html=True)
        st.markdown("<ul style='margin:0 0 0 18px; color:var(--muted)'><li>Be specific in `Situation`</li><li>Short messages produce focused replies</li><li>Use Temperature to adjust creativity</li></ul>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div class='footer'>Built with ❤️ using Transformer Neural Networks</div>", unsafe_allow_html=True)
