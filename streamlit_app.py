import streamlit as st
import torch
import torch.nn as nn
import os

st.set_page_config(page_title="Next Word Predictor")

class NextToken(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size_1, hidden_size_2, activation):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size_1)
        self.lin3 = nn.Linear(hidden_size_1, vocab_size)
        self.activation = activation
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.lin1(x))
        x = self.dropout(x)
        x = self.lin3(x)
        return x

@st.cache_resource
def load_model(model_name):
    try:
        checkpoint = torch.load(f'models/model_{model_name}.pth', map_location='cpu')
        hidden_sizes = {
            "small": 256,
            "medium": 512, 
            "large": 1024
        }
        model = NextToken(
            block_size=checkpoint['block_size'],
            vocab_size=len(checkpoint['wtoi']),
            emb_dim=checkpoint['emb_dim'],
            hidden_size_1=hidden_sizes[model_name],
            hidden_size_2=0,
            activation=torch.tanh
        )
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model, checkpoint['wtoi'], checkpoint['itow'], checkpoint['block_size']
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None
        
def predict_next_words(model, wtoi, itow, block_size, context_words, num_words=10, temperature=1.0):
    context_indices = []
    for word in context_words.split():
        if word in wtoi:
            context_indices.append(wtoi[word])
        else:
            context_indices.append(wtoi['#'])
    
    if len(context_indices) < block_size:
        context_indices = [wtoi['#']] * (block_size - len(context_indices)) + context_indices
    else:
        context_indices = context_indices[-block_size:]
    
    generated = context_words.split()
    
    with torch.no_grad():
        for _ in range(num_words):
            context_tensor = torch.tensor([context_indices])
            logits = model(context_tensor)
            logits = logits / temperature
            probs = torch.softmax(logits[0], dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            next_word = itow[next_idx]
            
            generated.append(next_word)
            context_indices = context_indices[1:] + [next_idx]
            
            if next_word in ['.', '!', '?']:
                break
    
    return ' '.join(generated)

# Streamlit UI
st.title("Next Word Predictor!")
st.write("Generate text using AI models trained on the lengendary writer Leo Tolstoy's Novel - War and Peace!")

# Sidebar controls
st.sidebar.header("Model Configuration")

model_variant = st.sidebar.selectbox(
    "Model Variant",
    ["small", "medium", "large"],
    help="Small: faster, less accurate | Large: slower, more accurate"
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Lower = more predictable+repetitive, Higher = more creative+gibberish"
)
num_words = st.slider("Words to generate", 1, 50, 15)

# Load selected model
model, wtoi, itow, block_size = load_model(model_variant)

# Check if model loaded successfully
if model is None or wtoi is None:
    st.error("Model files not found or failed to load. Please train the models first.")
    st.stop()

# Display model info
st.sidebar.info(f"""
**Model Info:**
- Context window: {block_size} words
- Embedding dim: {model.emb.embedding_dim}
""")

# Main input area
st.header("Text Input")
user_input = st.text_area(
    "Enter your text prompt:",
    value="The prince said",
    height=100,
    help=f"Model will use last {block_size} words as context"
)

if st.button("Generate Text", type="primary"):
    if user_input.strip():
        with st.spinner("Generating..."):
            try:
                result = predict_next_words(
                    model, wtoi, itow, block_size,
                    user_input, num_words, temperature
                )
                
                st.header("Generated Text")
                st.write(result)
                
                # Show context used
                context_words = user_input.split()[-block_size:]
                st.caption(f"Context used: {' '.join(context_words)}")
                
            except Exception as e:
                st.error(f"Error during generation: {str(e)}")
    else:
        st.warning("Please enter some text first")

# Model comparison info
st.sidebar.header("About Model Variants")
st.sidebar.info("""
**Small**: 3-word context, 32-dim embeddings  
**Medium**: 5-word context, 64-dim embeddings  
**Large**: 8-word context, 128-dim embeddings

All models trained on Tolstoy's War and Peace
""")
