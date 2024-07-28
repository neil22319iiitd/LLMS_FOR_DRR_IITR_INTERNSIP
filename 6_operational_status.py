import csv
import torch
import os
from transformers import LlamaTokenizer, LlamaForSequenceClassification
import numpy as np

# Set Hugging Face token
os.environ["HF_TOKEN"] = "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"

# Ensure the model runs on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the LLaMA-2-13B-hf model and tokenizer
model_name = "meta-llama/Llama-2-13b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name, pad_token="<pad>")
model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=len(['open', 'closed', 'partially open', 'partially closed', 'unknown'])).to(device)
print(f"Model {model_name} loaded successfully!")

# Define labels (adjust these based on your specific task)
OPERATIONAL_LABELS = ['open', 'closed', 'partially open', 'partially closed', 'unknown']
IMPACT_LABELS = ['blocked', 'blown', 'buried', 'burnt', 'collapsed', 'cracked', 'damaged', 'destroyed', 'displaced', 'disrupted', 'eroded', 'failed', 'flooded', 'ground liquefaction', 'ground shake', 'leakage', 'muddy', 'power outage', 'ruptured', 'slippery', 'torn', 'unsafe', 'uprooted', 'washed away', 'weakened', 'not_applicable']
SEVERITY_LABELS = ['severe', 'mild', 'moderate', 'unknown']

# Keywords for severity classification
SEVERITY_KEYWORDS = {
    'severe': ['severe', 'devastating', 'critical', 'catastrophic', 'extreme'],
    'moderate': ['moderate', 'significant', 'considerable', 'substantial'],
    'mild': ['mild', 'minor', 'slight', 'minimal']
}

def classify_severity(tweet):
    tweet_lower = tweet.lower()
    severity_scores = {label: 0 for label in SEVERITY_LABELS}

    for severity, keywords in SEVERITY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in tweet_lower:
                severity_scores[severity] += 1

    # Select the severity with the highest score
    max_score = max(severity_scores.values())
    if max_score > 0:
        return max(severity_scores, key=severity_scores.get)
    else:
        return 'unknown'

# Function to classify a tweet
def classify_tweet(tweet):
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    operational_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    # Print the softmax probabilities for debugging
    print(f"Softmax probabilities for tweet: {operational_probs}")
    
    # Determine operational status
    max_prob = np.max(operational_probs)
    operational_status_index = np.argmax(operational_probs)
    
    # Implement a threshold for uncertainty handling
    threshold = 0.5
    if max_prob < threshold:
        operational_status = 'unknown'
    else:
        operational_status = OPERATIONAL_LABELS[operational_status_index]
    
    # Manual checks for certain keywords to enhance accuracy
    tweet_lower = tweet.lower()
    if 'closed' in tweet_lower or 'not operating' in tweet_lower or 'shutdown' in tweet_lower:
        operational_status = 'closed'
    elif 'partially open' in tweet_lower or 'unsafe' in tweet_lower:
        operational_status='partially closed'
    elif 'partially open' in tweet_lower or 'partially operating' in tweet_lower:
        operational_status = 'partially open'
    elif 'open' in tweet_lower and 'partially' not in tweet_lower:
        operational_status = 'open'
    
    # Determine impact label based on keywords
    impact_label = 'not_applicable'  # Default to not_applicable if no relevant keywords are found
    for keyword in IMPACT_LABELS:
        if keyword in tweet_lower:
            impact_label = keyword
            break
    
    # Determine severity
    severity = classify_severity(tweet)
    
    return operational_status, impact_label, severity

# Load tweets from file
tweets_file = "week7_test_file.txt"
with open(tweets_file, "r", encoding="utf-8") as file:
    tweets = file.readlines()

# Open CSV file for writing
output_csv = "week7_llama_2_13B_3.csv"
with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Tweet", "Operational Status", "Impact Label", "Severity"])
    
    # Classify each tweet
    for tweet in tweets:
        tweet = tweet.strip()
        print(f"Classifying tweet: {tweet}")
        
        # Classify tweet using LLaMA-2-13B-hf
        operational_status, impact_label, severity = classify_tweet(tweet)
        
        # Write to CSV
        writer.writerow([tweet, operational_status, impact_label, severity])

print("Classification completed and saved to CSV.")
