import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import pickle
import math
import os

# ===============================================================
# PAGE CONFIGURATION
# ===============================================================

st.set_page_config(
    page_title="Empathetic Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
* {
    color: white !important;
}
body {
    background-color: #0D1117;
    color: white;
}
.stApp {
    background-color: #0D1117;
    color: white;
}
section[data-testid="stSidebar"] {
    background-color: #0D1117 !important;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
.sidebar .sidebar-content {
    background-color: #0D1117 !important;
}
.stButton>button {
    background-color: #238636;
    color: white !important;
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #2ea043;
    color: white !important;
}
.stTextInput>div>div>input {
    background-color: #0D1117;
    color: white !important;
    border-radius: 10px;
    border: 1px solid #30363D;
}
.stTextInput>div>div>input::placeholder {
    color: #8b949e !important;
}
.stTextArea textarea {
    background-color: #0D1117;
    color: white !important;
    border-radius: 10px;
    border: 1px solid #30363D;
}
.stTextArea textarea::placeholder {
    color: #8b949e !important;
}
.stSelectbox>div>div>div {
    background-color: #0D1117 !important;
}
.stSelectbox>div>div>div>div {
    background-color: #0D1117 !important;
    color: white !important;
}
.stSelectbox>div>div>div>button {
    background-color: #0D1117 !important;
    color: white !important;
}
[data-baseweb="select"] {
    background-color: #0D1117 !important;
}
[data-baseweb="select"] div {
    background-color: #0D1117 !important;
    color: white !important;
}
[data-baseweb="popover"] {
    background-color: #161B22 !important;
}
[data-baseweb="popover"] * {
    background-color: #161B22 !important;
    color: white !important;
}
.stSlider>div>div>div>div {
    color: white;
}
h1, h2, h3, h4, h5, h6 {
    color: white !important;
}
label {
    color: white !important;
}
.stMetric {
    color: white !important;
}
.stMetric>div {
    color: white !important;
}
.stMetric>div>div {
    color: white !important;
}
.stInfo {
    color: white !important;
}
.stSuccess {
    color: white !important;
}
.stWarning {
    color: white !important;
}
.stError {
    color: white !important;
}
.chat-container {
    border-radius: 15px;
    padding: 1em;
    background-color: #161B22;
    margin-bottom: 1em;
    color: white;
}
.chat-container * {
    color: white !important;
}
div[data-testid="stMarkdownContainer"] {
    color: white !important;
}
div[data-testid="stMarkdownContainer"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ===============================================================
# TRANSFORMER MODEL (from your notebook)
# ===============================================================

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

    def make_subsequent_mask(self, size):
        mask = torch.tril(torch.ones(size, size)).bool()
        return mask

    def forward(self, src, tgt):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = self.make_subsequent_mask(tgt.size(1))

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

# ===============================================================
# LOAD MODEL AND VOCABULARY
# ===============================================================

@st.cache_resource
def load_model_and_vocab():
    try:
        # Try to load vocabulary
        vocab_path = "vocab.pkl"
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
        
        with open(vocab_path, "rb") as f:
            vocab_data = pickle.load(f)

        # Extract word2idx and idx2word
        if isinstance(vocab_data, dict):
            if "word2idx" in vocab_data:
                word2idx = vocab_data["word2idx"]
                idx2word = vocab_data.get("idx2word", {v: k for k, v in word2idx.items()})
            else:
                word2idx = vocab_data
                idx2word = {v: k for k, v in word2idx.items()}
        else:
            raise ValueError("Unexpected vocab format")

        # Initialize model
        vocab_size = len(word2idx)
        pad_idx = word2idx.get("<pad>", 0)
        
        model = TransformerChatbot(
            vocab_size=vocab_size,
            d_model=256,
            num_heads=2,
            num_layers=2,
            dropout=0.1,
            pad_idx=pad_idx
        )

        # Try to load trained weights
        model_path = "best_model.pt"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            status = "✅ Model loaded successfully!"
        else:
            st.warning(f"Model weights not found at {model_path}. Using untrained model.")
            status = "⚠️ Using untrained model (weights not found)"

        model.eval()
        return model, word2idx, idx2word, status

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, f"❌ Error: {e}"

# Load model
model, word2idx, idx2word, model_status = load_model_and_vocab()

if model is None:
    st.stop()

# ===============================================================
# HELPER FUNCTIONS
# ===============================================================

def encode_text(text, word2idx, max_len=64):
    tokens = text.lower().split()
    ids = [word2idx.get(tok, word2idx.get("<unk>", 3)) for tok in tokens]
    ids = ids[:max_len - 2]
    ids = [word2idx.get("<bos>", 1)] + ids + [word2idx.get("<eos>", 2)]
    pad_len = max_len - len(ids)
    ids += [word2idx.get("<pad>", 0)] * pad_len
    return torch.tensor(ids).unsqueeze(0)

def decode_ids(ids, idx2word, word2idx):
    pad_idx = word2idx.get("<pad>", 0)
    bos_idx = word2idx.get("<bos>", 1)
    eos_idx = word2idx.get("<eos>", 2)
    tokens = [idx2word.get(i, "<unk>") for i in ids if i not in [pad_idx, bos_idx, eos_idx]]
    return " ".join(tokens)

def generate_reply_greedy(emotion, situation, customer_text, max_len=40):
    src = f"emotion: {emotion} | situation: {situation} | customer: {customer_text}"
    src_tensor = encode_text(src, word2idx)
    tgt_input = torch.tensor([[word2idx.get("<bos>", 1)]])

    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            output = model(src_tensor, tgt_input)
            logits = output[:, -1, :]
            next_token = logits.argmax(dim=-1)
            tgt_input = torch.cat([tgt_input, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == word2idx.get("<eos>", 2):
                break

    return decode_ids(tgt_input[0].tolist(), idx2word, word2idx)

def generate_reply_topk(emotion, situation, customer_text, max_len=40, temperature=0.8, top_k=20):
    src = f"emotion: {emotion} | situation: {situation} | customer: {customer_text}"
    src_tensor = encode_text(src, word2idx)
    tgt_input = torch.tensor([[word2idx.get("<bos>", 1)]])

    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            output = model(src_tensor, tgt_input)
            logits = output[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            top_k_val = min(top_k, probs.size(-1))
            top_probs, top_indices = torch.topk(probs, top_k_val)
            next_token = top_indices[0, torch.multinomial(top_probs[0], 1)]
            tgt_input = torch.cat([tgt_input, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == word2idx.get("<eos>", 2):
                break

    return decode_ids(tgt_input[0].tolist(), idx2word, word2idx)

def generate_reply_beam_search(emotion, situation, customer_text, max_len=40, beam_width=3):
    """Beam search decoding for higher quality responses"""
    src = f"emotion: {emotion} | situation: {situation} | customer: {customer_text}"
    src_tensor = encode_text(src, word2idx)
    bos_idx = word2idx.get("<bos>", 1)
    eos_idx = word2idx.get("<eos>", 2)
    pad_idx = word2idx.get("<pad>", 0)
    vocab_size = len(word2idx)

    model.eval()
    with torch.no_grad():
        # Initialize beam: (log_prob, sequence)
        beams = [(0.0, [bos_idx])]
        completed = []

        for step in range(max_len):
            candidates = []

            for log_prob, seq in beams:
                tgt_tensor = torch.tensor([seq])
                output = model(src_tensor, tgt_tensor)
                logits = output[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)[0]

                # Get top beam_width candidates
                top_log_probs, top_indices = torch.topk(log_probs, min(beam_width, vocab_size))

                for i, idx in enumerate(top_indices):
                    new_seq = seq + [idx.item()]
                    new_log_prob = log_prob + top_log_probs[i].item()
                    candidates.append((new_log_prob, new_seq))

            # Sort by log probability and keep top beam_width
            candidates.sort(reverse=True, key=lambda x: x[0])
            beams = []

            for log_prob, seq in candidates[:beam_width]:
                if seq[-1] == eos_idx:
                    completed.append((log_prob, seq))
                else:
                    beams.append((log_prob, seq))

            if not beams:
                break

        # Return best completed sequence or best incomplete
        if completed:
            best_seq = max(completed, key=lambda x: x[0])[1]
        else:
            best_seq = max(beams, key=lambda x: x[0])[1] if beams else [bos_idx]

    return decode_ids(best_seq, idx2word, word2idx)

def generate_reply(emotion, situation, customer_text, decoding="greedy", max_len=40, temperature=0.8, beam_width=3, top_k=20):
    if decoding == "greedy":
        return generate_reply_greedy(emotion, situation, customer_text, max_len)
    elif decoding == "topk":
        return generate_reply_topk(emotion, situation, customer_text, max_len, temperature, top_k)
    elif decoding == "beam_search":
        return generate_reply_beam_search(emotion, situation, customer_text, max_len, beam_width)
    else:
        return generate_reply_greedy(emotion, situation, customer_text, max_len)

# ===============================================================
# UI
# ===============================================================

st.sidebar.title("🤖 Empathetic Chatbot")
st.sidebar.markdown(model_status)
st.sidebar.divider()
st.sidebar.info("Your AI companion trained to respond with empathy 💖")

st.title("💬 Empathetic Chatbot")
st.markdown("Ask, vent, or share — this chatbot understands your emotions")

emotion = st.selectbox("Emotion", ["sad", "happy", "angry", "anxious", "lonely", "grateful", "hopeful"])
situation = st.text_input("Describe your situation", "lost my favorite item")
customer_text = st.text_area("What would you like to say?", "I can't find it anywhere, I feel terrible.")

st.markdown("---")
st.subheader("⚙️ Generation Settings")

col1, col2, col3 = st.columns(3)
with col1:
    decoding = st.selectbox(
        "Decoding Strategy",
        ["greedy", "topk", "beam_search"],
        help="🟢 Greedy: Fast, picks highest probability\n🟡 Top-K: Balanced, diverse\n🔵 Beam Search: Slower, highest quality"
    )
with col2:
    max_len = st.slider("Max Length", 20, 100, 50, help="Maximum tokens in response")
with col3:
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, step=0.1, help="Higher = more creative")

# Conditional settings based on decoding strategy
if decoding == "beam_search":
    st.info("🔵 **Beam Search** explores multiple sequences simultaneously for better quality responses", icon="ℹ️")
    beam_width = st.slider("Beam Width", 2, 10, 3, help="Number of beams (higher = better but slower)")
    top_k = 20
    temperature = 0.8
elif decoding == "topk":
    st.info("🟡 **Top-K Sampling** randomly picks from top K candidates for diversity", icon="ℹ️")
    beam_width = 3
    top_k = st.slider("Top-K", 5, 50, 20, help="Sample from top K tokens")
else:
    st.info("🟢 **Greedy Decoding** selects the highest probability token at each step", icon="ℹ️")
    beam_width = 3
    top_k = 20

if st.button("💬 Generate Response", use_container_width=True):
    with st.spinner("Thinking empathetically..."):
        response = generate_reply(
            emotion, 
            situation, 
            customer_text, 
            decoding=decoding, 
            max_len=max_len, 
            temperature=temperature,
            beam_width=beam_width,
            top_k=top_k
        )
    
    st.success("Here's what I'd say:")
    st.markdown(f"""
    <div class="chat-container">
        <b>🧍 You:</b> {customer_text}<br><br>
        <b>🤖 Chatbot:</b> {response}
    </div>
    """, unsafe_allow_html=True)
    
    # Show generation stats
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    with col_stats1:
        st.metric("Method", decoding.upper())
    with col_stats2:
        st.metric("Response Length", len(response.split()))
    with col_stats3:
        st.metric("Temperature", temperature)

st.markdown("---")
st.caption("Built with PyTorch + Streamlit | Empathetic Chatbot")