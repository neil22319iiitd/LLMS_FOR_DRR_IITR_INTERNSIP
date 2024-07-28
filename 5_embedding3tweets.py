import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

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

# Define a function to generate an embedding for a single tweet
def generate_embedding(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]  # Get the last layer hidden state
    embedding = hidden_states.mean(dim=1).cpu().numpy().flatten()
    return embedding

# Three tweets for which to generate the embeddings
tweet1 = "Thrikkakara Municipal Co-operative Hospital is dedicated to providing excellent healthcare services to the community with commitment and care. #Healthcare #CommunityCare #Thrikkakara" 
tweet2="Thrikkakara Municipal Co-operative Hospital faced significant challenges during the Kerala floods of 2018, with floodwaters causing disruptions to essential healthcare services. #KeralaFloods #FloodImpact #Thrikkakara"
tweet3 = "The disaster impact on Thrikkakara Municipal Co-operative Hospital during the flood was substantial, affecting vital healthcare services. #DisasterImpact #Healthcare #Flood"



# Generate the embeddings
embedding1 = generate_embedding(tweet1, model, tokenizer, device)
embedding2 = generate_embedding(tweet2, model, tokenizer, device)
embedding3 = generate_embedding(tweet3, model, tokenizer, device)

# Save the embeddings to separate CSV files, overwriting any existing data
output_embeddings_file1 = 'embedding1_week6.csv'
output_embeddings_file2 = 'embedding2_week6.csv'
output_embeddings_file3 = 'embedding3_week6.csv'

pd.DataFrame([embedding1]).to_csv(output_embeddings_file1, mode='w', index=False, header=False)
pd.DataFrame([embedding2]).to_csv(output_embeddings_file2, mode='w', index=False, header=False)
pd.DataFrame([embedding3]).to_csv(output_embeddings_file3, mode='w', index=False, header=False)

print(f"Embedding saved to {output_embeddings_file1}")
print(f"Embedding saved to {output_embeddings_file2}")
print(f"Embedding saved to {output_embeddings_file3}")
