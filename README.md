# NanoHelp GPT

A small GPT model that I built and trained completely from scratch in PyTorch. It reads a customer support message and writes a support style reply, all from a transformer I wrote myself and trained on my own cleaned dataset.

This is not ChatGPT. It is a 12.3M parameter model trained for a few minutes on a free Colab GPU. The point of this project was never to match a huge model, it was to actually understand and build every piece of a GPT myself: the tokenizer, the attention heads, the training loop, and the debugging that comes with all of it.

**Try it live:** [add your Hugging Face Spaces link here]
**Code:** this repo

---

## What it does

You type a customer message like

```
I need help cancelling my order
```

and the model generates a support agent style reply, token by token, using attention it learned entirely from data. No pretrained weights were used anywhere. Everything, from the tokenizer to the final layer, was trained from zero.

---

## Why I built this

I wanted to actually understand how a GPT works instead of just calling an API. Karpathy's video is a great base for the architecture, so I did not reinvent that part. What makes this project mine is everything around it: the dataset I picked and cleaned, the custom tokenizer I trained, the model size decisions, and a training process that genuinely went wrong a few times before it worked. I kept that whole story in this README because figuring out why a model sounds broken is honestly the most useful part of a project like this.

---

## The dataset

I used Bitext's customer support dataset from Hugging Face: `bitext/Bitext-customer-support-llm-chatbot-training-dataset`.

* 26,872 customer message and agent reply pairs
* 27 different support intents (cancelling orders, refunds, account issues, shipping, and so on)
* Licensed under cdla sharing 1.0
* It is synthetic data, not real company tickets. I am saying that clearly here because it matters. It is fine for a learning project like this, but it is also why the replies sometimes sound a bit templated.

### Cleaning and formatting

I did not use the raw dataset as is. Here is what I changed:

1. Collapsed extra whitespace and newlines in both the customer message and the agent response.
2. Dropped any response longer than 1000 characters. That removed about 12 percent of the rows but kept every example short enough to fit comfortably inside the model's context window.
3. Kept the `{{Order Number}}` style placeholders exactly as they are. About half the responses contain at least one. Instead of stripping them, I treat them as single tokens during tokenization and swap in realistic fake values at the demo stage, like `ORD 48213`.
4. Kept the typos in customer messages on purpose, things like "cancelling puchase" are what real customers actually type.
5. No missing values and no duplicate rows in the original dataset.

Every row got reformatted into one training string:

```
Customer: <message>
Agent: <reply><END>
```

I split the data 90 percent train, 10 percent validation, by row, before training the tokenizer, so nothing from validation ever leaked into training.

---

## The tokenizer

I trained my own byte level BPE tokenizer on this data instead of using something off the shelf like GPT 2's tokenizer or tiktoken. A few reasons:

* GPT 2's vocabulary has over 50,000 tokens. The embedding table for that alone would be bigger than my entire transformer.
* My own tokenizer, trained just on this support data, only needs a vocab of 4096 tokens, which keeps the whole model small and fast to train.
* I made `<END>` and every `{{placeholder}}` type its own single token, so the model can never generate a broken, half finished placeholder tag.

After cleaning, the whole dataset came out to about 2.59M training tokens and 290K validation tokens with this vocabulary. I verified the tokenizer perfectly round trips text back and forth (encode then decode gives you the exact original string back) on 500 random samples before trusting it for training.

---

## The model

A decoder only GPT, built layer by layer in PyTorch, following the architecture from Karpathy's video:

* Multi head self attention with causal masking, so the model can only look at earlier tokens, never future ones
* A feed forward layer after attention in every block
* Pre norm residual connections (layernorm before each sublayer, not after)
* Weight tying between the input embedding table and the output layer

**Config:**

| | |
|---|---|
| Embedding size | 384 |
| Attention heads | 6 |
| Layers | 6 |
| Context length | 256 tokens |
| Vocabulary | 4096 |
| Dropout | 0.3 |
| **Total parameters** | **12.3M** |

Trained on a free Google Colab T4 GPU.

---

## How training actually went (the part I almost left out)

I want to be upfront about this because it is the most useful part of the whole project. My first few training attempts did not work, and figuring out why taught me more than the parts that went smoothly.

**Attempt 1:** I trained for 3000 steps and watched the loss drop all the way down to basically zero on both train and validation. Looked perfect on paper. Then I actually read the generated text and it was pure garbage, the model would just repeat one word over and over, things like "ETA ETA ETA ETA" forever.

The reason took me a while to understand properly. Loss is measured in a mode called teacher forcing, where the model is shown the correct previous tokens at every step and only graded on predicting the next one. But when you actually generate text, the model only sees its own previous outputs, and any small wobble early on compounds into nonsense. A model with near zero loss is extremely overconfident, and once it drifts even slightly off the exact training pattern, it has no idea what to do and just collapses into repeating itself. Low loss does not mean good generation. That became the central lesson of this whole project.

**Attempt 2 and 3:** I also found a dumb but real bug. I was setting `dropout = 0.3` as a line of code written after the model was already built, which does nothing, because the layers read that value only once, when they are constructed. So an entire training run quietly used the wrong dropout the whole time without throwing any error. Once I fixed the order (set dropout before building the model), training behaved differently.

I also lowered the learning rate, since my first attempts were almost certainly too aggressive for a model and dataset this size.

**Fixing generation, not just training:** even after fixing dropout and the learning rate, the model would still sometimes slip into repeating single words or short phrases during generation. I fixed this on the generation side, not the training side, by writing a sampling function that does two things at once:

* A mild penalty on any token the model has already used, so it is less likely to pick the exact same word again
* A hard block on repeating any three token sequence it has already generated

That combination is what finally stopped the loops completely.

**Picking a checkpoint by actually reading it, not by loss:** I saved a checkpoint every 50 training steps instead of only keeping whatever had the lowest validation loss. Then I went through several checkpoints by hand, generating real replies from each one and reading them, because the checkpoint with the lowest loss was not the one that sounded the best. That is a strange thing to learn firsthand, but it is true, and it is exactly the kind of thing a loss curve alone will never tell you.

The final clean training run, after all these fixes, trained smoothly for 700 steps with no restarts and no repetition collapse, and I picked the checkpoint by comparing generated text across steps 200 through 700 directly.

---

## Sampling settings

Once training was solid, I compared a few temperature and top k combinations on the same prompts:

| Setting | Temperature | Top K | Result |
|---|---|---|---|
| Tight | 0.4 | 10 | Most coherent, a bit short |
| Medium | 0.6 | 20 | Good balance |
| Loose | 0.9 | 50 | More varied but noticeably rougher |

I settled on **temperature 0.5, top k 15** as the default for the demo.

---

## What the model is actually good and bad at

Being honest about failure cases is a strength here, not something to hide, so here is a real look at both.

**Where it does well:** the model has clearly learned the shape of a support reply. It opens politely, gives numbered steps when appropriate, and closes naturally. It even handled a prompt I made up that is nothing like the training data, "my dog ate my invoice, what do I do," by picking up on the word "invoice" and giving genuinely relevant steps. That is real generalization, not memorization.

**Where it struggles:** sometimes the reply is fluent and well formed, but answers the wrong question. For example, asking "how do I reset my account password" sometimes returns steps for updating a shipping address instead. The sentence structure is learned very well, but with 27 different intents and a model this size trained on a fairly small, repetitive dataset, it has not fully learned to tell every intent apart. It tends to fall back on a few frequently seen templates (shipping address steps showed up a lot) rather than always matching the right one. That is a specific, explainable limitation, not just "the model is bad sometimes."

---

## How to replicate this yourself

If you want to run this whole pipeline from scratch, here is the order I did it in. Everything was run in Google Colab on a free T4 GPU.

### 1. Get and clean the data

* Load `bitext/Bitext-customer-support-llm-chatbot-training-dataset` with the `datasets` library
* Drop responses over 1000 characters
* Format every row as `Customer: ...\nAgent: ...<END>`
* Split 90/10 into train and validation

### 2. Train the tokenizer

* Train a byte level BPE tokenizer (I used the `tokenizers` library) on the training split only
* Vocab size 4096, with `<END>` and every `{{placeholder}}` added as special tokens
* Encode both splits and save them as `train.bin` and `val.bin`

### 3. Build the model

* A decoder only transformer in PyTorch: causal self attention, multi head attention, feed forward blocks, pre norm residual connections, weight tying
* Config used here: 384 embedding size, 6 heads, 6 layers, 256 context length, 0.3 dropout

### 4. Train

* AdamW optimizer, cosine learning rate schedule with warmup, peak learning rate 3e-4
* **Set dropout before building the model, not after**
* Save a checkpoint every 50 steps, and actually read the generated text at each checkpoint instead of only trusting the loss number
* Pick the checkpoint that reads best, not the one with the lowest loss

### 5. Generate text properly

* Use a sampling function that combines a repetition penalty with hard n gram blocking, otherwise the model will eventually loop on a word or phrase
* Temperature 0.5, top k 15 worked best here

### 6. Deploy

* Wrap the model in a Gradio interface
* Swap `{{placeholder}}` tokens for realistic fake values at generation time, so the demo looks natural
* Deploy the Space to Hugging Face, CPU is enough for a model this size

---

#

## Credits

* Architecture approach based on Andrej Karpathy's "Let's build GPT" video and nanoGPT
* Dataset: Bitext customer support dataset (cdla sharing 1.0 license), synthetic data generated by Bitext, not real customer tickets
* Built and trained independently in PyTorch on Google Colab
