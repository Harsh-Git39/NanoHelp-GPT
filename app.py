import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from tokenizers import Tokenizer

# ---------- config (must match training exactly) ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.3
block_size = 256

tok = Tokenizer.from_file("tokenizer.json")
vocab_size = tok.get_vocab_size()

# ---------- model definition ----------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------- load trained weights ----------
model = GPT().to(device)
model.load_state_dict(torch.load("best.pt", map_location=device))
model.eval()

# ---------- generation ----------
@torch.no_grad()

def reply(prompt, max_new=60, temperature=0.5, top_k=15, rep_penalty=1.3, no_repeat_ngram=3):

    ids = tok.encode("Customer: " + prompt + "\nAgent:").ids
    x = torch.tensor([ids], device=device)

    for _ in range(max_new):

        logits, _ = model(x[:, -block_size:])
        logits = logits[0, -1, :].clone() / temperature

        
        for tid in set(x[0].tolist()):
            logits[tid] /= rep_penalty
        seq = x[0].tolist()
        if len(seq) >= no_repeat_ngram - 1:
            banned = set()
            prefix = tuple(seq[-(no_repeat_ngram - 1):])
            for i in range(len(seq) - no_repeat_ngram + 1):
                if tuple(seq[i:i + no_repeat_ngram - 1]) == prefix:
                    banned.add(seq[i + no_repeat_ngram - 1])
            for tid in banned:
                logits[tid] = float("-inf")
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        x = torch.cat([x, next_id.unsqueeze(0)], dim=1)
    text = tok.decode(x[0].tolist(), skip_special_tokens=False)
    return text.split("<END>")[0].split("Agent:")[-1].strip()


# ---------- fake placeholder values for a realistic demo ----------
FAKE_VALUES = {
    "Order Number": lambda: f"ORD-{random.randint(10000,99999)}",
    "Account Type": lambda: random.choice(["Premium", "Standard", "Business"]),
    "Account Category": lambda: random.choice(["Personal", "Business"]),
    "Customer Support Phone Number": lambda: "1-800-555-0199",
    "Customer Support Hours": lambda: "Mon-Fri, 9am-6pm",
    "Website URL": lambda: "www.example.com/support",
    "Invoice Number": lambda: f"INV-{random.randint(100000,999999)}",
    "Tracking Number": lambda: f"TRK-{random.randint(100000000,999999999)}",
    "Date Range": lambda: "3-5",
}


def fill_placeholders(text):
    def sub(m):
        key = m.group(1)
        return FAKE_VALUES.get(key, lambda: key)()
    return re.sub(r"\{\{(.*?)\}\}", sub, text)


def generate_response(prompt, max_tokens, temperature, top_k):
    if not prompt.strip():
        return "Please enter a customer inquiry."
    raw = reply(prompt.strip(), max_new=int(max_tokens), temperature=float(temperature), top_k=int(top_k))
    return fill_placeholders(raw)


demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(lines=3, placeholder="Type a customer question here...", label="Customer Query"),
        gr.Slider(minimum=20, maximum=150, value=60, step=10, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.05, label="Temperature"),
        gr.Slider(minimum=1, maximum=50, value=15, step=1, label="Top-K"),
    ],
    outputs=gr.Textbox(lines=5, label="Support-GPT Response"),
    title="Support-GPT (12.3M Parameter Transformer, trained from scratch)",
    description="A GPT trained from scratch on synthetic customer-support data. Not ChatGPT quality, this is a small, from-scratch model, and that's the point.",
    examples=[
        ["I need help cancelling my order", 60, 0.5, 15],
        ["how do I check the status of my refund", 60, 0.5, 15],
        ["I want to update my shipping address", 60, 0.5, 15],
        ["my dog ate my invoice, what do I do", 60, 0.5, 15],
    ],
)

if __name__ == "__main__":
    demo.launch()
