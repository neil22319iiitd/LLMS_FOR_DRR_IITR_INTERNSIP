import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

# Load the tweets from the text file
tweets_file = '4_tweetfile.txt'

with open(tweets_file, 'r', encoding='utf-8') as file:
    tweets = [line.strip() for line in file.readlines()]

# Access token for Hugging Face (if necessary)
access_token = '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'

# Initialize LLaMA2-13B tokenizer and model
model_name = "meta-llama/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

# Add a padding token if it does not exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)

# Ensure the model is set to evaluation mode
model.eval()

# Check if CUDA is available and move model to GPU
device = torch.device("cuda")
model.to(device)

# Define a function to generate embeddings
def generate_embeddings_batched(texts, model, tokenizer, device, batch_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} / {len(texts)//batch_size + 1}")
        inputs = tokenizer(batch, return_tensors='pt', max_length=512, truncation=True, padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Get the last layer hidden state
        batch_embeddings = hidden_states.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    return embeddings

# Generate embeddings for tweets
embeddings = generate_embeddings_batched(tweets, model, tokenizer, device, batch_size=8)

# Save the embeddings to a CSV file
output_embeddings_file = '4_llama2_13b_embeddings.csv'
pd.DataFrame(embeddings).to_csv(output_embeddings_file, index=False, header=None)

print(f"Embeddings saved to {output_embeddings_file}")
