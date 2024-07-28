import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Define the impact labels
impact_labels = [
    "blocked", "burnt", "collapsed", "cracked", "damaged", "destroy", 
    "displaced", "eroded", "failed", "flooded", "leakage", "poweroutage", 
    "ruptured", "slippery", "torn", "tornoff", "incapable", "unsafe", 
    "uprooted", "weakened", "block","damage","keralafloods",
]

# Initialize a counter for the impact labels and unknown category
impact_counter = Counter({label: 0 for label in impact_labels})
impact_counter['unknown'] = 0

# Read tweets from the file
with open('week6_manual_tweets.txt', 'r', encoding='utf-8') as file:
    tweets = file.readlines()

# Process each tweet
for tweet in tweets:
    tweet = tweet.lower()
    words = tweet.split()
    hashtags = [word for word in words if word.startswith('#')]
    for hashtag in hashtags:
        label = hashtag.lstrip('#')
        if label in impact_labels:
            impact_counter[label] += 1
        else:
            impact_counter['unknown'] += 1

# Print the counts
for label, count in impact_counter.items():
    print(f"#{label}: {count}")

# Prepare data for the heatmap
labels = list(impact_counter.keys())
counts = list(impact_counter.values())

# Create a matrix for the heatmap
heatmap_data = [[count] for count in counts]

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="viridis", xticklabels=["Count"], yticklabels=labels)
plt.title('Impact Labels Hashtag Count')
plt.xlabel('Count')
plt.ylabel('Impact Labels')
plt.show()
