import streamlit as st
import tiktoken
from core import *
import torch
import torch.nn.functional as F

if 'numbrow' not in st.session_state:
    st.session_state['numbrow'] = 500 
if 'maxlen' not in st.session_state:
    st.session_state['maxlen'] = 32
if 'batches' not in st.session_state:
    st.session_state['batches'] = 4
if 'Epochs' not in st.session_state:
    st.session_state['Epochs'] = 30

st.set_page_config(
    page_title="GPT2 Architecture Explainer & Implementation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title('GPT - Understanding & Implementation [Pytorch]')

sides = [
    "What & Why",
    "Evolution",
    "Architecture", 
    "From Pretrained (GPT-2)",
    "Train Own Data",
    "Compare with BiLSTM/RNN",
    "About App"
]
add_selectbox = st.sidebar.selectbox(
    "Navigation", tuple(sides), key='navigation_selectbox'
)


if add_selectbox == "What & Why":
    st.header("What is the purpose of this page?")
    st.write("This is a conceptual and implementational explainer on what GPTs are, how they work and why are they such an important milestone in modelling large langiage models.")
    st.write("The flow will be as follows:\n1. Evolution of GPTs\n2. Its's architechture\n3. Loading weights from pre-trained gpt2\n4. Training our own next token prediction model\n5. Comparision with best NON-TRASNFORMER model")
   

    st.divider()
    st.subheader("Begin!")
    st.divider()
    
    st.caption("So Mr __Latest__ and *Best* GPT 4o... Tell me something about :blue[yourself].")

        




if add_selectbox  == "Evolution":
    st.header("Theory")







if add_selectbox == "Architecture":
    st.write("""
            Generative Pretrained Transformers are a decoder only abstraction of the famous Transformers architechtire, specifically designed for next token prediction.\
            """)
    st.image("sgsr.jpg", caption = "Transformers vs GPT")
    

    st.markdown("""**Generative**: They were designed primarily for autoregressive tasks like next token prediction. This is why the encoder block is completely removed along with the cross attention block, which is now replaced with a masked self attention block.
                **Pre-trained**: GPT models are pre-trained on extensive corpora of text from the internet. This pre-training allows the model to learn a wide range of language patterns, facts, and even some forms of reasoning.
                **Transformers**: They are adapted from the transformers architechtures with the following changes:

    1. Removal of encoder block.\n
    2. Removal of cross attention block.\n
    3. Layer norm was shifted to before MLP FFN.\n
    4. Additional layer norm was added after final self attention block.\n
    """)

    st.divider()
    st.subheader("The GPT configuration class")
    st.write("The first piece of code. This a dataclass to store the GPT model configurations.")
    st.code("""from dataclasses import dataclass
@dataclass
class GPTConfiguration:
            '''Configurations as in GPT-2'''
    block_size: int = 1024  
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True
""")
    with st.expander("Block Size"):
        st.code("block_size: int = 1024")
        st.write("The block size determines the maximum length of the input sequence that the model can process at one time. In GPT-2, the block size is 1024 tokens, meaning the model can consider up to 1024 tokens at a time when generating text. This large context window allows GPT-2 to generate coherent and contextually appropriate text over relatively long passages.")

    with st.expander("Vocabulary Size"):
        st.code("vocab_size: int = 50257")
        st.write("The vocabulary size of GPT-2 is 50257, which includes all unique tokens that the model can recognize. These tokens are created using byte pair encoding (BPE), which allows the model to work with subword units, enabling it to handle a wide range of language and tokenization scenarios, including uncommon words and multi-word expressions.")

    with st.expander("Number of Layers"):
        st.code("n_layer: int = 12")
        st.write("GPT-2 consists of 12 transformer layers (also known as decoder blocks). Each layer includes multiple attention heads and feedforward neural networks that process and transform the input sequence. The depth of these layers allows GPT-2 to learn complex patterns and relationships within the data, contributing to its ability to generate sophisticated and contextually relevant text.")

    with st.expander("Number of Attention Heads"):
        st.code("n_head: int = 12")
        st.write("GPT-2 uses 12 attention heads per layer. Each attention head processes the input from a different perspective, allowing the model to capture various relationships within the data. This multi-head attention mechanism enables GPT-2 to focus on different parts of the input sequence simultaneously, improving its ability to understand and generate complex text.")

    with st.expander("Embedding Size"):
        st.code("n_embd: int = 768")
        st.write("The embedding size in GPT-2 is 768, which defines the dimensionality of the token embeddings and hidden states in the model. Each token in the input sequence is represented as a 768-dimensional vector. This dimensionality is crucial for capturing detailed information about each token and plays a significant role in the model's overall performance and capacity.")

    with st.expander("Dropout Rate"):
        st.code("dropout: float = 0.1")
        st.write("GPT-2 uses a dropout rate of 0.1 as a regularization technique to prevent overfitting. During training, 10% of the neurons are randomly dropped out, encouraging the model to generalize better to unseen data. This slight dropout helps the model avoid becoming too reliant on any particular feature, which can improve its robustness and performance on diverse tasks.")

    with st.expander("Bias Usage"):
        st.code("bias: bool = True")
        st.write("In GPT-2, bias terms are used in the linear transformations within the model. These bias terms allow the model to shift the activation functions, providing greater flexibility in fitting the training data. Bias terms are included in the model's architecture to enhance learning and improve the model's ability to capture complex patterns in the data.")


    st.divider()

    st.subheader("The GPT Class")
    st.write("The GPT Class, inherits from the torch.nn.Module class, is a construction class to create a GPT model based on the configurations")

    st.code("""
    class GPT2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wtp = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            """)
    
    with st.expander("__init__ method - Initialize the GPT-2 model"):
        st.code("""
class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        """)
        st.write("This is the constructor of the `GPT2` class, which is a subclass of `nn.Module`, making it a PyTorch neural network module. The `__init__` method is used to initialize the model's parameters. The `config` parameter holds the configuration settings, such as the number of layers, embedding size, etc. The `super().__init__()` call initializes the parent `nn.Module` class. The `self.config` attribute stores the configuration for later use.")

    with st.expander("Define Transformer Components using nn.ModuleDict"):
        st.code("""
    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd),
        wtp = nn.Embedding(config.block_size, config.n_embd),
        drop = nn.Dropout(config.dropout),
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f = nn.LayerNorm(config.n_embd),))
        """)
        st.write("The `self.transformer` attribute is defined as a `ModuleDict`, which is a PyTorch dictionary that can store various sub-modules. This dictionary contains the components of the Transformer model:\n\n"
                "- `wte`: The word token embedding layer, which maps token indices (from the input sequence) into dense vectors of size `n_embd`. It uses the vocabulary size `vocab_size` and embedding dimension `n_embd`.\n"
                "- `wtp`: The positional embedding layer, which assigns a unique position embedding to each position in the sequence up to `block_size`. This is also an embedding layer of size `n_embd`.\n"
                "- `drop`: A dropout layer, which is applied to the embeddings and activations to prevent overfitting during training. The dropout rate is specified by `config.dropout`.\n"
                "- `h`: A list of `Block` layers, where each block represents a transformer layer. The number of layers is determined by `n_layer`, and the `Block` class defines the internals of each transformer block.\n"
                "- `ln_f`: A layer normalization layer applied to the final output of the transformer before passing it to the output layer. Layer normalization helps stabilize the learning process.")

    with st.expander("Language Model Head"):
        st.code("""
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        """)
        st.write("The `self.lm_head` is a linear layer that maps the final hidden states from the transformer (with dimensionality `n_embd`) to the vocabulary size (`vocab_size`). This layer produces logits for each token in the vocabulary, which are used to predict the next token in the sequence. The `bias=False` indicates that no bias term is used in this linear transformation, simplifying the model slightly.")

    with st.expander("Forward Method - Forward Pass through the Model"):
        st.code("""
    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward, because {len(idx)} > {config.block_size}"
        """)
        st.write("The `forward` method defines the forward pass through the model, where `idx` is the input tensor containing token indices. The method first extracts the batch size `B` and sequence length `T` from the shape of `idx`. An assertion is included to ensure that the sequence length `T` does not exceed the model's maximum block size (`config.block_size`). If the input sequence is too long, the assertion triggers an error.")

    with st.expander("Compute Token and Positional Embeddings"):
        st.code("""
    #pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
    tok_emb = self.transformer.wte(idx)
    pos_emb = self.transformer.wtp(torch.arange(T))
    x = tok_emb + pos_emb
        """)
        st.write("In this block:\n"
                "- `tok_emb`: The model computes the token embeddings by passing the input indices (`idx`) through the word token embedding layer `wte`. This converts the token indices into dense vectors.\n"
                "- `pos_emb`: The positional embeddings are computed by generating a range of positions from 0 to `T-1` (the sequence length) and passing these positions through the positional embedding layer `wtp`.\n"
                "- `x = tok_emb + pos_emb`: The token and positional embeddings are summed to produce the input to the transformer layers. This sum combines the token-specific information with the positional context, allowing the model to understand the order of tokens in the sequence.")

    with st.expander("Apply Dropout to the Embeddings"):
        st.code("""
    x = self.transformer.drop(x)
        """)
        st.write("The combined embeddings are passed through the dropout layer (`self.transformer.drop`). Dropout randomly zeroes some of the elements of the input tensor during training, which helps prevent overfitting by making the model less sensitive to specific features of the data.")

    with st.expander("Pass Through Each Transformer Block"):
        st.code("""
    for block in self.transformer.h:
        x = block(x)
        """)
        st.write("The input `x` is sequentially passed through each of the transformer blocks stored in `self.transformer.h`. Each block processes the input, applying attention mechanisms and feedforward neural networks to transform the input sequence. The output of each block becomes the input to the next block. This repeated transformation allows the model to learn complex representations of the input data.")

    with st.expander("Apply Final Layer Normalization"):
        st.code("""
    x = self.transformer.ln_f(x)
        """)
        st.write("After passing through all the transformer blocks, the final output `x` is normalized using the layer normalization layer `ln_f`. Layer normalization helps stabilize the learning process by normalizing the output of each sequence element to have a consistent mean and variance. This step prepares the output for the final linear transformation.")

    with st.expander("Compute Logits for Each Token"):
        st.code("""
    logits = self.lm_head(x)
        """)
        st.write("The normalized output `x` is passed through the language model head (`self.lm_head`), which is a linear layer that maps the embeddings to the size of the vocabulary. This produces a tensor of logits, where each element corresponds to the model's prediction for the next token in the sequence. These logits are used in the next step to compute probabilities and generate text or evaluate predictions.")

    with st.expander("Return the Logits"):
        st.code("""
    return logits
        """)
        st.write("Finally, the logits tensor is returned as the output of the `forward` method. These logits can be used to generate text by selecting the highest-probability token at each step or by sampling from the probability distribution defined by the logits. In training, the logits are typically passed to a loss function, which compares them to the true token indices to compute the model's loss.")

        
    st.divider()
    st.subheader("The attention Block class")
    st.write("This class handles the implementation of the attention block, of which there are 12. This involves causal self attention, multi layered perceptron and residual connections")
    st.code("""
class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)
  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
""")
    with st.expander("__init__ method - Initialize the Transformer Block"):
        st.code("""
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    """)
        st.write("The `Block` class defines a single transformer block, which is a fundamental component of the GPT model. This block consists of a self-attention mechanism followed by a feedforward neural network, with layer normalization applied before each of these components. The `__init__` method initializes these components based on the configuration provided (`config`):\n\n"
             "- `self.ln_1`: This is a layer normalization layer applied to the input before the self-attention mechanism. It helps stabilize and normalize the input, improving the learning process.\n"
             "- `self.attn`: This is an instance of the `CausalSelfAttention` class, which implements the self-attention mechanism with causality (ensuring that each token can only attend to previous tokens, not future ones). This attention mechanism is crucial for capturing dependencies between tokens in the sequence.\n"
             "- `self.ln_2`: This is another layer normalization layer, applied before the feedforward network. It ensures that the input to the MLP is well-normalized, aiding in training stability.\n"
             "- `self.mlp`: This is an instance of the `MLP` class, representing the feedforward neural network that processes the output of the attention mechanism. The MLP typically consists of two linear layers with a non-linearity in between.")

    with st.expander("Forward Method - Forward Pass through the Transformer Block"):
        st.code("""
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        """)
        st.write("The `forward` method defines how data flows through the transformer block. It processes the input `x` through the self-attention mechanism and the feedforward network, with residual connections to maintain the flow of information:\n\n"
                "- `x = x + self.attn(self.ln_1(x))`: The input `x` is first normalized using `self.ln_1`. The normalized output is then passed through the `self.attn` (self-attention) mechanism. The result of the self-attention is added back to the original input `x` through a residual connection. This residual connection helps mitigate the vanishing gradient problem and allows the model to learn more effectively by preserving the original input information.\n"
                "- `x = x + self.mlp(self.ln_2(x))`: Similarly, the output from the previous step is normalized again using `self.ln_2` and passed through the `self.mlp` (multi-layer perceptron) network. The result is then added back to the input using another residual connection. This step allows the model to apply more complex transformations to the data while still preserving the original information.\n"
                "- `return x`: The final transformed `x` is returned as the output of the block. This output can then be passed to the next block in the model, allowing the entire transformer to build complex, layered representations of the input data.")

    with st.expander("Layer Normalization (ln_1 and ln_2)"):
        st.code("""
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.ln_2 = nn.LayerNorm(config.n_embd)
        """)
        st.write("Layer normalization (`ln_1` and `ln_2`) is applied to the input before the self-attention and MLP components. Layer normalization normalizes the input across the feature dimension, ensuring that the mean and variance are consistent across layers. This helps stabilize and speed up the training process, as it reduces internal covariate shift. By applying layer normalization before the self-attention and MLP layers, the model ensures that the input to these layers is well-behaved, which improves overall model performance.")

    with st.expander("Causal Self-Attention (attn)"):
        st.code("""
    self.attn = CausalSelfAttention(config)
        """)
        st.write("The `attn` component is an instance of the `CausalSelfAttention` class, which implements the self-attention mechanism with a causal mask. This mask ensures that each token can only attend to previous tokens (and not future ones), preserving the autoregressive property necessary for tasks like language modeling. Self-attention allows the model to focus on different parts of the input sequence when making predictions, capturing dependencies across the sequence regardless of distance. This is a key part of what makes transformer models like GPT effective at understanding and generating coherent text.")

    with st.expander("Feedforward Network (mlp)"):
        st.code("""
    self.mlp = MLP(config)
        """)
        st.write("The `mlp` component is an instance of the `MLP` class, which implements a feedforward neural network. Typically, this network consists of two linear layers with a non-linear activation function (such as GELU) in between. The MLP applies further transformations to the data after self-attention, allowing the model to capture more complex patterns and interactions in the data. The MLP is a crucial component for enhancing the model's expressive power, enabling it to model complex relationships in the input data.")

    st.divider()
    st.subheader("The MulitLayerPerceptron class")
    st.write("handles the individual token mapping to output layer")
    st.code("""
class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
    self.gelu = nn.GELU()
    self.dropout = nn.Dropout(config.dropout)
  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    x = self.dropout(x)
    return
""")
    with st.expander("__init__ method - Initialize the MLP (Multi-Layer Perceptron)"):
        st.code("""
    class MLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(config.dropout)
        """)
        st.write("The `MLP` class defines the feedforward network used within each transformer block in the GPT model. It consists of two linear transformations with a non-linear activation function in between, followed by dropout for regularization. The `__init__` method initializes these components based on the provided configuration (`config`):\n\n"
                "- `self.c_fc`: This is the first linear layer, which projects the input from the original embedding size (`n_embd`) to a higher dimensional space (`4 * n_embd`). The factor of 4 is commonly used to increase the model's capacity to capture more complex relationships.\n"
                "- `self.c_proj`: This is the second linear layer, which projects the output from the higher dimensional space back down to the original embedding size (`n_embd`). This layer effectively compresses the expanded representation, allowing the model to transform the data in a non-linear fashion.\n"
                "- `self.gelu`: This is the GELU (Gaussian Error Linear Unit) activation function, which introduces non-linearity after the first linear transformation. GELU is chosen for its smooth activation curve, which often leads to better performance in deep learning models compared to ReLU or other activation functions.\n"
                "- `self.dropout`: This dropout layer is applied after the second linear transformation to prevent overfitting during training. The dropout rate is specified by `config.dropout`, controlling the fraction of neurons that are randomly set to zero during training.")

    with st.expander("Forward Method - Forward Pass through the MLP"):
        st.code("""
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
        """)
        st.write("The `forward` method defines how data flows through the MLP. This method processes the input `x` through the linear layers, the activation function, and finally the dropout layer:\n\n"
                "- `x = self.c_fc(x)`: The input `x` is passed through the first linear layer `c_fc`, which projects it from its original size (`n_embd`) to a higher-dimensional space (`4 * n_embd`). This expansion allows the model to capture more complex features.\n"
                "- `x = self.gelu(x)`: The output of the first linear layer is then passed through the GELU activation function. This non-linear activation introduces non-linearity to the model, allowing it to learn more complex patterns in the data.\n"
                "- `x = self.c_proj(x)`: The activated output is passed through the second linear layer `c_proj`, which reduces the dimensionality back to the original embedding size (`n_embd`). This compression step transforms the expanded representation back into a form suitable for further processing in the transformer block.\n"
                "- `x = self.dropout(x)`: Finally, dropout is applied to the output of the second linear layer. This regularization step helps prevent overfitting by randomly setting a fraction of the activations to zero during training.\n"
                "- `return x`: The transformed and regularized output `x` is returned as the final result of the MLP. This output can then be used in the transformer block for further processing or passed to the next layer in the model.")

    with st.expander("First Linear Layer (c_fc)"):
        st.code("""
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        """)
        st.write("The `c_fc` layer is the first linear transformation in the MLP. It projects the input from its original embedding size (`n_embd`) to a higher-dimensional space (`4 * n_embd`). This expansion allows the model to increase its capacity to capture and represent more complex features. The `bias=config.bias` parameter indicates whether a bias term is included in this linear transformation, as specified in the configuration.")

    with st.expander("GELU Activation Function (gelu)"):
        st.code("""
    self.gelu = nn.GELU()
        """)
        st.write("The `gelu` layer applies the GELU (Gaussian Error Linear Unit) activation function to the output of the first linear layer. GELU is a smooth, non-linear activation function that is often preferred over ReLU in transformer models due to its ability to better model complex patterns. It allows the MLP to introduce non-linearity, enabling the model to learn more sophisticated relationships in the data.")

    with st.expander("Second Linear Layer (c_proj)"):
        st.code("""
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        """)
        st.write("The `c_proj` layer is the second linear transformation in the MLP. It projects the high-dimensional representation (`4 * n_embd`) back down to the original embedding size (`n_embd`). This compression step allows the model to refine and distill the expanded features into a form that can be effectively used in the next stages of the transformer block. As with the first linear layer, the presence of a bias term is controlled by the `config.bias` parameter.")

    with st.expander("Dropout Layer (dropout)"):
        st.code("""
    self.dropout = nn.Dropout(config.dropout)
        """)
        st.write("The `dropout` layer applies dropout regularization to the output of the second linear layer. Dropout helps prevent overfitting by randomly zeroing out a fraction of the neurons during training, which forces the model to learn more robust features. The dropout rate is specified by `config.dropout`, which determines the probability of dropping out each neuron.")


    st.divider()
    st.subheader("The CausalSelfAttention class")
    st.write("This class handles the masked multi head self attention")
    st.code("""
class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    self.attn_dropout = nn.Dropout(config.dropout)
    self.resid_dropout = nn.Dropout(config.dropout)  
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.dropout = config.dropout
    self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size,config.block_size))
  def forward(self, x):
    B, T, C = x.shape
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
    att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)
    y = att @ v
    y = y.transpose(1,2).contiguous().view(B, T, C)
    y = self.resid_dropout(self.c_proj(y))
    return y
""")

    with st.expander("__init__ method - Initialize the Causal Self-Attention Layer"):
        st.code("""
    class CausalSelfAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            assert config.n_embd % config.n_head == 0
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.attn_dropout = nn.Dropout(config.dropout)
            self.resid_dropout = nn.Dropout(config.dropout)  
            self.n_head = config.n_head
            self.n_embd = config.n_embd
            self.dropout = config.dropout
            self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size,config.block_size))
        """)
        st.write("The `CausalSelfAttention` class implements the self-attention mechanism with causality, which is critical for autoregressive models like GPT. The `__init__` method initializes the various components necessary for performing multi-head self-attention with a causal mask:\n\n"
                "- `assert config.n_embd % config.n_head == 0`: Ensures that the embedding size (`n_embd`) is divisible by the number of attention heads (`n_head`). This is necessary for splitting the embeddings across the attention heads.\n"
                "- `self.c_attn`: A linear layer that projects the input into three separate matrices: query (Q), key (K), and value (V), each of size `n_embd`. The output of this layer has a size of `3 * n_embd`.\n"
                "- `self.c_proj`: A linear layer that projects the concatenated output of the attention heads back to the original embedding size (`n_embd`). This layer is applied after the attention mechanism has been completed.\n"
                "- `self.attn_dropout`: Dropout applied to the attention weights to prevent overfitting. This layer is used after the softmax operation on the attention scores.\n"
                "- `self.resid_dropout`: Dropout applied to the output of the attention mechanism before adding the residual connection, helping to prevent overfitting.\n"
                "- `self.n_head`: Stores the number of attention heads, `n_head`, from the configuration.\n"
                "- `self.n_embd`: Stores the embedding size, `n_embd`, from the configuration.\n"
                "- `self.dropout`: Stores the dropout rate from the configuration.\n"
                "- `self.register_buffer('bias', ...)`: This creates a causal mask (a lower triangular matrix) to ensure that each token can only attend to itself and previous tokens, not future ones. This mask is essential for autoregressive tasks like language modeling.")

    with st.expander("Forward Method - Forward Pass through the Causal Self-Attention Layer"):
        st.code("""
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
        """)
        st.write("The `forward` method defines how the input `x` is processed through the causal self-attention mechanism. The steps are as follows:\n\n"
                "- `B, T, C = x.shape`: The input tensor `x` has a shape of (Batch size, Sequence length, Embedding size). These dimensions are extracted for further processing.\n"
                "- `q, k, v = self.c_attn(x).split(self.n_embd, dim=2)`: The input `x` is passed through the `c_attn` linear layer, which projects it into three separate matrices: query (`q`), key (`k`), and value (`v`). These are then split into separate tensors, each of size `n_embd`.\n"
                "- `k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)`: The key tensor `k` is reshaped to split it across the attention heads (`n_head`), creating a tensor of shape `(Batch size, n_head, Sequence length, Head size)`. The `transpose` function swaps the batch and head dimensions to facilitate parallel processing of attention heads.\n"
                "- `q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)`: Similarly, the query tensor `q` is reshaped and transposed to align with the multi-head attention format.\n"
                "- `v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)`: The value tensor `v` is also reshaped and transposed to align with the multi-head attention format.\n"
                "- `att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))`: The attention scores are computed by performing a matrix multiplication between the query `q` and the transpose of the key `k`. The result is scaled by the square root of the head size (to maintain stable gradients), producing the raw attention scores.\n"
                "- `att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))`: The attention scores are masked using the causal mask (`self.bias`) to ensure that each token only attends to itself and previous tokens. Future positions are masked out by setting their corresponding attention scores to `-inf`, preventing the model from attending to them.\n"
                "- `att = F.softmax(att, dim=-1)`: The masked attention scores are passed through a softmax function to convert them into probabilities, making the sum of the attention weights for each query equal to 1.\n"
                "- `att = self.attn_dropout(att)`: Dropout is applied to the attention weights to prevent overfitting, ensuring that the model does not become too reliant on any single part of the sequence.\n"
                "- `y = att @ v`: The attention weights are used to compute a weighted sum of the value vectors `v`, producing the context vectors `y`.\n"
                "- `y = y.transpose(1,2).contiguous().view(B, T, C)`: The context vectors `y` are transposed back to the original shape `(Batch size, Sequence length, Embedding size)`, making them ready for the next stage in the transformer block.\n"
                "- `y = self.resid_dropout(self.c_proj(y))`: The context vectors are passed through the `c_proj` linear layer, which projects them back to the original embedding size. Dropout is applied to the output before adding it to the residual connection, helping to prevent overfitting.\n"
                "- `return y`: The final output `y`, which includes the self-attention and residual processing, is returned for further processing in the transformer block.")

    with st.expander("Query, Key, and Value Projections (c_attn)"):
        st.code("""
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        """)
        st.write("The `c_attn` layer is a linear transformation that projects the input tensor into three distinct components: query (`q`), key (`k`), and value (`v`). These components are essential for computing the self-attention mechanism. The layer's output has three times the embedding size (`3 * n_embd`) because it concatenates the query, key, and value vectors into a single tensor, which is then split in the `forward` method. The `bias` parameter controls whether bias terms are added to the linear transformations.")

    with st.expander("Causal Mask Initialization (register_buffer)"):
        st.code("""
    self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size,config.block_size))
        """)
        st.write("This line initializes a lower triangular matrix (`torch.tril`) as a buffer named `bias`, which is used to create the causal mask. The causal mask ensures that the model only attends to current and previous positions in the sequence, never to future positions. This is crucial for maintaining the autoregressive property required in language models like GPT. The buffer is shaped to match the attention mechanism's expected input dimensions and is applied during the computation of attention scores.")

    with st.expander("Attention Scores Computation"):
        st.code("""
    att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        """)
        st.write("The attention scores are computed by performing a matrix multiplication between the query (`q`) and the transpose of the key (`k`). This operation measures the similarity between each query and key pair, which is then scaled by the square root of the dimensionality of the key vectors. Scaling the scores helps maintain stable gradients and avoids extremely large values that could destabilize the learning process. These scores are the raw attention weights before applying the softmax function and the causal mask.")

    with st.expander("Apply Causal Mask to Attention Scores"):
        st.code("""
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        """)
        st.write("This line applies the causal mask to the attention scores. The `masked_fill` function replaces elements of the attention matrix corresponding to future positions (as indicated by the causal mask) with negative infinity (`-inf`). This ensures that when the softmax function is applied later, these positions will have an attention weight of zero, effectively preventing the model from attending to any tokens beyond the current position in the sequence.")

    with st.expander("Softmax and Dropout on Attention Weights"):
        st.code("""
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)
        """)
        st.write("The `att` tensor, which contains the masked attention scores, is passed through the softmax function to convert it into a probability distribution. This operation ensures that the attention weights for each query sum to one, allowing the model to focus on the most relevant parts of the input sequence. After the softmax operation, dropout (`self.attn_dropout`) is applied to the attention weights as a regularization technique, reducing the likelihood of overfitting by randomly setting some of the attention weights to zero during training.")

    with st.expander("Compute Context Vectors and Apply Residual Dropout"):
        st.code("""
    y = att @ v
    y = y.transpose(1,2).contiguous().view(B, T, C)
    y = self.resid_dropout(self.c_proj(y))
        """)
        st.write("In this part of the forward method:\n\n"
                "- `y = att @ v`: The attention weights (`att`) are multiplied with the value vectors (`v`) to compute the context vectors `y`. This operation aggregates the information from different parts of the input sequence, weighted by their relevance as determined by the attention mechanism.\n"
                "- `y = y.transpose(1,2).contiguous().view(B, T, C)`: The context vectors `y` are transposed and reshaped to their original format of `(Batch size, Sequence length, Embedding size)`, making them compatible with the next layers in the model.\n"
                "- `y = self.resid_dropout(self.c_proj(y))`: The context vectors are then passed through the `c_proj` linear layer, which projects them back to the original embedding size. Dropout (`resid_dropout`) is applied after this projection to further regularize the model by randomly dropping out some of the activations, helping to prevent overfitting.")



if add_selectbox == "From Pretrained (GPT-2)":
    st.divider()
    st.header("Loading weigts from gpts for inference")
    st.write("This section will explore the inference capabilities of the gpt2 architechture by \
             loading in the weights from hugging face 'GPT2LMHeadModel'")
    
    st.write("We are going to add this method to the main GPT class")

    st.code("""
@classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024 

                    config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return mode
                """)
    
    with st.container(border=True):
        st.subheader("Code explaination")
        with st.expander("Class Method - from_pretrained"):
            st.code("""
        @classmethod
        def from_pretrained(cls, model_type):
            """)
            st.write("The `from_pretrained` method is defined as a class method using the `@classmethod` decorator. This means the method belongs to the class itself (`cls`), rather than to instances of the class. The method's purpose is to load a pretrained GPT-2 model's weights from Hugging Face's model hub, specified by the `model_type` parameter. The method returns an instance of the class (`cls`) with the loaded weights.")

        with st.expander("Check Model Type"):
            st.code("""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
            """)
            st.write("This line ensures that the `model_type` provided is one of the supported GPT-2 model variants: `gpt2`, `gpt2-medium`, `gpt2-large`, or `gpt2-xl`. If the provided `model_type` is not in this set, the method raises an assertion error, preventing further execution.")

        with st.expander("Import GPT2LMHeadModel from Transformers"):
            st.code("""
        from transformers import GPT2LMHeadModel
            """)
            st.write("This line imports the `GPT2LMHeadModel` class from the Hugging Face `transformers` library. This class is used to load the pretrained GPT-2 model from Hugging Face's model hub, which will be used to initialize the weights of the custom GPT model.")

        with st.expander("Print Model Loading Message"):
            st.code("""
        print("loading weights from pretrained gpt: %s" % model_type)
            """)
            st.write("This line prints a message to the console, indicating that the model weights are being loaded from the specified pretrained GPT-2 model (`model_type`). This is useful for debugging and tracking the progress of the model initialization.")

        with st.expander("Define Model Configuration Arguments"):
            st.code("""
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
            """)
            st.write("This block defines the configuration arguments (`config_args`) for different GPT-2 model types. Depending on the `model_type` provided, the appropriate configuration is selected. Each configuration specifies the number of layers (`n_layer`), attention heads (`n_head`), and embedding size (`n_embd`) for the model. The comments indicate the number of parameters in each model type. The `config_args` dictionary for the selected model type is then used to configure the custom GPT model.")

        with st.expander("Set Vocabulary and Block Size"):
            st.code("""
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
            """)
            st.write("These lines set the `vocab_size` and `block_size` for the model configuration. The `vocab_size` is fixed at 50257, which is the vocabulary size for all GPT-2 models, and the `block_size` is set to 1024, indicating the maximum sequence length the model can handle. These values are standard for GPT-2 and are necessary for ensuring compatibility with the pretrained weights.")

        with st.expander("Create GPT Configuration and Model"):
            st.code("""
        config = GPTConfig(**config_args)
        model = GPT(config)
            """)
            st.write("These lines create a configuration object (`config`) for the custom GPT model using the `config_args` defined earlier. The `GPTConfig` class takes these arguments and configures the model accordingly. Then, a new instance of the custom GPT model (`model`) is created using this configuration. This model is initialized from scratch, meaning its weights are not yet loaded with the pretrained values.")

        with st.expander("Get Model State Dictionary"):
            st.code("""
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
            """)
            st.write("The `state_dict` of the custom GPT model is retrieved and stored in `sd`. The `state_dict` is a dictionary containing all the model's parameters (weights and biases) and buffers. The keys of this dictionary (`sd_keys`) represent the names of these parameters. The subsequent line filters out keys that end with `.attn.bias`, as these are masks or buffers that do not need to be matched with the pretrained model's parameters.")

        with st.expander("Load Pretrained Hugging Face Model"):
            st.code("""
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
            """)
            st.write("These lines load the pretrained GPT-2 model from Hugging Face using the specified `model_type`. The `GPT2LMHeadModel.from_pretrained()` method downloads the model and initializes it with the pretrained weights. The `state_dict` of this pretrained model (`sd_hf`) is then retrieved, which will be used to copy the weights into the custom GPT model.")

        with st.expander("Filter Pretrained Model State Dictionary Keys"):
            st.code("""
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
            """)
            st.write("These lines filter the keys from the pretrained model's `state_dict` (`sd_hf`). The first filter removes keys that end with `.attn.masked_bias`, and the second filter removes keys that end with `.attn.bias`. Both of these keys correspond to buffers (not learnable parameters) used for masking during the attention mechanism. Since these buffers are not parameters, they do not need to be copied into the custom model.")

        with st.expander("Transpose Specific Weights for Compatibility"):
            st.code("""
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
            """)
            st.write("This line defines a list of specific weight names (`transposed`) that require special treatment. These weights correspond to layers that, in the original OpenAI implementation, use a `Conv1D` module. However, in the custom GPT implementation, these layers are implemented as vanilla linear layers (`nn.Linear`). Therefore, the weights need to be transposed to be compatible with the custom model's implementation.")

        with st.expander("Ensure Parameter Alignment and Copy Weights"):
            st.code("""
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
            """)
            st.write("These lines ensure that the number of keys (parameters) in the pretrained model's state dictionary (`sd_keys_hf`) matches the number of keys in the custom GPT model's state dictionary (`sd_keys`). If they do not match, an assertion error is raised.\n\n"
                    "The code then iterates over each key in `sd_keys_hf`:\n\n"
                    "- If the key corresponds to one of the layers that require transposition (`transposed` list), the shape of the pretrained weight is checked against the shape of the custom model's weight. The shape of the pretrained weight is reversed (`[::-1]`) to account for the transposition. If the shapes match, the weight is transposed (`.t()`) and copied into the custom model.\n"
                    "- If the key does not require transposition, the shapes of the weights in the pretrained and custom models are directly compared. If they match, the weight is copied over without transposition.\n\n"
                    "The `torch.no_grad()` context manager is used to prevent PyTorch from tracking these operations for gradient computation, as this is a manual parameter initialization step, not a part of the training process.")

        with st.expander("Return the Custom Model"):
            st.code("""
        return model
            """)
            st.write("The final line of the method returns the custom GPT model (`model`) with the pretrained weights successfully loaded. This model can now be used for further tasks, such as fine-tuning or generating text.")


    with st.container(border=True):
        st.subheader("Loading weights and running inference")
        st.write("The 'from_pretrained' is a class method or constructor which can construct the GPT class with the given cofig.")
        
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }
        for mod in config_args.keys():
            st.markdown(f"**Model:** `{mod}`      \
                        **Params:** {config_args[mod]}")
        gpt2_config = list(config_args.keys())
        selection = st.selectbox("Select the GPT2 config (n_layer, n_head, n_embd)", gpt2_config)
        if st.button(f"Load Weights for {selection}", use_container_width=True):
            with st.status("Loading weights"):
                st.write("Importing module...")
                
                st.write("Downloading weights...")
        model = GPT.from_pretrained(selection)
        st.code(model.config)
    st.subheader("Forwarding the model")
    st.write("For generation, we will set number of return samples and max number of token generation as our parameters (just like huggingface pipeline).")
    "First we put our model into evaluation mode. This is done because much of the layer like dropouts and batch normalisations are not required during inference.\nDo this using `model.eval()`"
    num_return_sequence = st.slider("Number of return sequences", 3, 18, 5)
    max_length = st.slider("Max length of a return sequence", 15, 50, 30)
    
    "Since GPU is not available at this moment, we do not need to send the tensors to cuda using `model.to('cuda')`"
    url = "https://tiktokenizer.vercel.app"

    st.write("Our input will first be tokenized using the gpt2 tokenizer using the tiktokenizer library. The tokens can be viewed [here](%s)" % url)
    ptp = 'Hi I am GPT2, and I can'
    "Our input prompt will be 'Hi I am GPT2, and I can'."
    "Or you can write your own prompt below"
    ppp = st.text_input("Enter Prompt", ptp)

    with st.status("Tokenizing..."):
        
        tk = tiktoken.encoding_for_model('gpt2')
        token = tk.encode(ppp)
        st.write(token)
        st.write("Converting to tensors...")
        token = torch.tensor(token, dtype=torch.long)
        st.write(token)
        token = token.unsqueeze(0).repeat(num_return_sequence, 1)
        st.write(token)
        x = token

    st.subheader("Sampling and inference")
    "Set manual seed for consistency over iterations: "
    seed = st.number_input("Enter manual seed value", 0, 1000, 42)
    while x.size(1) < max_length:
        with torch.no_grad():    
            logits = model(x)
            #st.write(type(logits))
            #st.write(logits)
            logits = logits[0]
            logits = logits[:, -1, :]
            probs = F.softmax(logits,dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 20, dim = -1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)
    for i in range(num_return_sequence):
        token = list(x[i, :max_length])
        decoded = tk.decode(token)
        st.write(">", decoded)



if add_selectbox == "Train Own Data":
    st.header("Train on your own data")
    st.write("Here you can either use the default Shakespeare dataset or upload your own .txt file to train GPT from scratch.")

    tab1, tab2 = st.tabs(['Shakespeare', 'Custom Data'])
    
    with tab1:
        st.subheader("Training with Shakespeare Data")
        if st.button("Get Shakespeare Data", use_container_width=True):
            with st.spinner("Getting Shakespeare data..."):
                text = download_shakespeare_data()
                st.success("Data ready!")
            st.session_state['numbrow'] = st.number_input("Enter the number of characters to display", 10, 2000, 100)
            to_view = text[:st.session_state['numbrow']]
            with st.expander(f"View the first {st.session_state['numbrow']} characters of your dataset"):
                st.markdown(to_view)
            tokenizer = tiktoken.get_encoding('gpt2')
            encoded_sample = {}
            sam_tok = tokenizer.encode(to_view)
            
            with st.expander("Explore tokens: "):
                for i, j in zip(to_view[:100], sam_tok[:100]):
                    st.write(f"**Word:** '{i}', **Token:** {j}")

            
            st.session_state['maxlen'] = st.number_input("Enter the max length of a sequence", 5, 176, 32)
            st.session_state['batches'] = st.number_input("Enter the number of batches", 1, 10, 4)

            st.write("Create tensor with selected length and batches.")
            k, p = st.session_state['maxlen'], st.session_state['batches']
            st.write(f"Length of sequence: {k}, Number of batches: {p}, Number of tokens: {k * p + 1}")
            train = tokenizer.encode(text)[:k * p + 1]
            with st.expander(f"View the first {k*p+1} tokens of your dataset"):
                st.write(train)
            tenr = torch.tensor(train)
            st.write("Getting inputs and outputs")
            x = tenr[:-1].view(p, k)
            y = tenr[1:].view(p, k)
            with st.expander("View reshaped inputs and targets of first batch"):
                st.write("inputs")
                st.write(x[0,:])
                st.write("targets")
                st.write(y[0,:])
            
            model = GPT(GPTConfig())
            logits, _ = model(x)
            st.write(logits.shape)
            optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1, lr=6e-4, betas=(0.9, 0.95))
            st.write("We will overfit a single batch, by repeatedly training data for 50 iterations.\
                This will overfit the batch.")
            if st.button(f"Sample Training with subset of dataset, upto {k*p} tokens", use_container_width=True):
                with st.status("View iterations"):
                    for i in range(50):
                        st.write("Optimizer zero grad")
                        optimizer.zero_grad()
                        st.write("Get logits and loss")
                        logits, loss = model(x, y)
                        st.write("Backward pass")
                        loss.backward()
                        st.write("Optimize")
                        optimizer.step()
                        st.write(f'Loss at `{i}` iteration: `{loss.item()}`')

            st.write("Load entire data with Dataloader")
            train_loader = Dataloader(B=st.session_state['batches'], T=st.session_state['maxlen'])
            st.write("Run training loop for entire data...")
            st.session_state['Epochs'] = st.number_input("Enter the number of epochs", 10, 100, 30)

            if st.button("Full Train"):
                for i in range(st.session_state['Epochs']):
                    x, y = train_loader.next_batch()
                    optimizer.zero_grad()
                    logits, loss = model(x, y)
                    loss.backward()
                    optimizer.step()
                    st.write("step ", i, " loss: ", loss.item())

            seed = "This is my"

            st.write("### Evaluation with Seed Text")
            seed_text = st.text_input("Enter Prompt", "This is my")
            st.write(f"Seed text: {seed_text}")

            num_return_sequence = st.number_input("Enter the number of sequences to generate", 1, 10, 1)
            max_length = st.number_input("Enter the max length of generated sequence", 10, 200, 50)

            with st.spinner("Tokenizing..."):
                tokenizer = tiktoken.encoding_for_model('gpt2')
                token = tokenizer.encode(seed_text)
                st.write("Tokenized input:", token)

                token = torch.tensor(token, dtype=torch.long)
                st.write("Tensor:", token)

                token = token.unsqueeze(0).repeat(num_return_sequence, 1)
                st.write("Tensor for batch generation:", token)
                x = token

            st.subheader("Sampling and Inference")
            st.write("Set manual seed for consistency over iterations:")
            seed = st.number_input("Enter manual seed value", 0, 1000, 42)
            torch.manual_seed(seed)

            while x.size(1) < max_length:
                with torch.no_grad():    
                    logits = model(x)
                    logits = logits[0][:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 20, dim=-1)
                    ix = torch.multinomial(topk_probs, 1)
                    next_token = torch.gather(topk_indices, -1, ix)
                    x = torch.cat((x, next_token), dim=1)

            for i in range(num_return_sequence):
                token = list(x[i, :max_length])
                decoded = tokenizer.decode(token)
                st.write(f"Generated Sequence {i+1}: {decoded}")

    with tab2:
        st.subheader("Upload Your Own Dataset")
        file = st.file_uploader("Upload your text file", type=["txt"])
        if file is not None:
            st.success("File uploaded successfully!")
            text = file.read().decode("utf-8")
            st.session_state['numbrow'] = st.number_input("Enter the number of characters to display", 10, 2000, 100)
            to_view = text[:st.session_state['numbrow']]
            with st.expander(f"View the first {st.session_state['numbrow']} characters of your dataset"):
                st.markdown(to_view)
            tokenizer = tiktoken.get_encoding('gpt2')
            encoded_sample = tokenizer.encode(to_view)

            st.write("Tokenized data:")
            st.write(encoded_sample)

            st.write("Create (Batch, Sequence) tokens from the token list above")
            buf = torch.tensor(encoded_sample)
            with st.expander("View tensor"):
                st.write(buf)
            st.write("Resizing into (B,T) tensor")
            st.session_state['maxlen'] = st.number_input("Enter the max length of a sequence", 5, 176, 32)
            st.session_state['batches'] = st.number_input("Enter the number of batches", 1, 10, 4)

            k, p = st.session_state['maxlen'], st.session_state['batches']
            train = tokenizer.encode(text)[:k * p + 1]
            tenr = torch.tensor(train)
            x = tenr[:-1].view(p, k)
            y = tenr[1:].view(p, k)

            st.write("Input tensor:")
            st.write(x)

            st.write("Target tensor:")
            st.write(y)

            model = GPT(GPTConfig())
            logits, _ = model(x)
            st.write(logits.shape)
            optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1, lr=6e-4, betas=(0.9, 0.95))
            if st.button("Sample Training"):
                for i in range(50):
                    optimizer.zero_grad()
                    logits, loss = model(x, y)
                    loss.backward()
                    optimizer.step()
                    st.write(f'Loss at {i} iteration: {loss.item()}')

            train_loader = Dataloader(B=st.session_state['batches'], T=st.session_state['maxlen'])
            st.write("Run training loop for entire data...")
            st.session_state['Epochs'] = st.number_input("Enter the number of epochs", 10, 100, 30)

            if st.button("Full Train"):
                for i in range(st.session_state['Epochs']):
                    x, y = train_loader.next_batch()
                    optimizer.zero_grad()
                    logits, loss = model(x, y)
                    loss.backward()
                    optimizer.step()
                    st.write("step ", i, " loss: ", loss.item())

            seed = "This is my"

            st.write("### Evaluation with Seed Text")
            seed_text = st.text_input("Enter Prompt", "This is my")
            st.write(f"Seed text: {seed_text}")

            num_return_sequence = st.number_input("Enter the number of sequences to generate", 1, 10, 1)
            max_length = st.number_input("Enter the max length of generated sequence", 10, 200, 50)

            with st.spinner("Tokenizing..."):
                tokenizer = tiktoken.encoding_for_model('gpt2')
                token = tokenizer.encode(seed_text)
                st.write("Tokenized input:", token)

                token = torch.tensor(token, dtype=torch.long)
                st.write("Tensor:", token)

                token = token.unsqueeze(0).repeat(num_return_sequence, 1)
                st.write("Tensor for batch generation:", token)
                x = token

            st.subheader("Sampling and Inference")
            st.write("Set manual seed for consistency over iterations:")
            seed = st.number_input("Enter manual seed value", 0, 1000, 42)
            torch.manual_seed(seed)

            while x.size(1) < max_length:
                with torch.no_grad():    
                    logits = model(x)
                    logits = logits[0][:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 20, dim=-1)
                    ix = torch.multinomial(topk_probs, 1)
                    next_token = torch.gather(topk_indices, -1, ix)
                    x = torch.cat((x, next_token), dim=1)

            for i in range(num_return_sequence):
                token = list(x[i, :max_length])
                decoded = tokenizer.decode(token)
                st.write(f"Generated Sequence {i+1}: {decoded}")


        
            
            


