This is an implementation of the approach described by the research paper https://arxiv.org/abs/2404.14432 (Monitoring Critical Infrastructure Facilities During Disasters Using Large Language Models)
The paper has taken Christchurch, New Zealand, susceptible to earthquakes, and Broward County, Florida, USA, susceptible to hurricanes, So the authors took them as The AOI(area of interest)
We have taken the Infamous kerela floods of 2018 which caused humungous damage in India.

1_OSM_CIF-
1.Setup and Initialization : I set up the necessary libraries and tools. I imported
requests for making HTTP requests, pandas for data manipulation, and
Nominatim from geopy.geocoders for geolocation services.
2.Define Area of Interest (AOI) AOI):I specified the area of interest (AOI) as Kerala,
India.
3.Retrieve Location Details : Using Nominatim , I fetched the latitude and
longitude coordinates of Kerala.
4.Search for Critical Infrastructure Facilities (CIFs) CIFs):I defined types of Critical
Infrastructure Facilities (CIFs) to search for, such as hospitals and fire stations.
5.Retrieve CIFs Using Overpass API :I created a function to query the Overpass
API to retrieve CIFs (like hospitals and fire stations) within a specified radius
around the coordinates of Kerala.
6.Data Handling and Storage :I collected the retrieved CIFs into a Data Frame
using pandas for easy manipulation and visualization. Finally, I saved the Data
Frame as a CSV file named 'kerala_cifs.csv
This code essentially automates the process of finding and collecting information about
hospitals and fire stations in Kerala, India, using geographic data and APIs
In The CIF files I have gathered around 708 CIF’s which were potentially affected
during the kerela floods 2018




2_WebScraping.Py-
Due to Twitter discontinuing free access to their API, I had to continue using my current web scraping code
with Selenium and Beautiful Soup. Here are the changes I have made to the code (` webscrapper.py
1. Keywords for Queries: I have compiled a list of over 100 keywords relevant to our study.
These keywords will be used to form different queries
2. Organizing Keywords: To optimize the running time of the code and avoid crashes, I have
stored the keywords in 20 separate files. This segmentation helps in managing the process more
efficiently and prevents the code from running for too long at a stretch, which can cause crashes and
result in tweets not being saved properly
3. Generating Search Queries: For each keyword, a query is created to generate a URL for
searching on Twitter. The general format of the query is: ` Kerala flood 2018 + keyword `. This ensures that
all tweets related to a specific keyword are extracted
4 . Combining Tweets: By combining all these files, I have gathered roughly 450 tweets.
CONCLUSION
: But I found out later that the tweets generated from this code is not of very good quality and
later cause problems because we need to have a large number of good quality tweets for our study which
actually has some useful messages for us . This would have been a good method if I had access to twitter
API key.




3_manual_tweets.txt -
The research paper discussed generating synthetic tweets using the LLAMA 2 13B model.
I attempted to generate tweets for 12 Critical Infrastructure Facilities (CIFs), sending 4
CIFs at a time and randomly generating 10 15 tweets for each. Despite batching the CIFs, it
took about 8 9 hours for the code to run
Given that OpenAI has discontinued the free use of the GPT 3 API, my next option was to
manually ask ChatGPT 3.5 to generate diverse tweets for each CIF. I randomly chose 114
CIFs from the 708 we had, aiming to match the roughly 110 120 CIFs used in our research
paper. My prompt to GPT 3.5 was: " Generate { num_tweets } diverse tweets describing the impact of the
Kerala floods on { cif_name } at cif_address }." The variable num_tweets varied between 5, 10, 15, and
sometimes 20. The CIF names and addresses were sourced from a CSV file
(kerala_cifs.csv
I successfully generated around 1140 high quality tweets for my study. Before generating
the tweets, I instructed GPT 3.5 to use 20 specific keywords in every tweet and include a
hashtag for each used keyword at the end of the tweet. The keywords were: "blocked",
"burnt", "collapsed", "cracked", "damaged", "destroyed", "displaced", "eroded", "failed",
"flooded", "leakage", "power outage", "ruptured", "slippery", "torn", "torn off", "incapable",
"unsafe", "uprooted", "weakened
I also had to manually adjust many tweets to ensure their relevance to our discussion.

4_llama2_embeddings.py- 
1.The next and main step of the implementation was using llama2 13b model to generate
embeddings for all the tweets(1140 in total) The llama 2 embeddings were of size 5120 each which means each embeddings formed a vectorof 5120 size and was stored in a csv file
This code also required a lot of changes and took roughly 1012 hrs to run even after creating batches (batch size 16)

Brief explanation of the code:
1.Step 1: Import Necessary Libraries : First , I imported the necessary libraries. I used torch to
work with PyTorch , transformers from the Hugging Face library to use pre trained language
models, and pandas to handle data in a structured format.
2.Step 2: Load Tweets from a Text File Next , I loaded the tweets from a text file named
week6_manual_tweet.txt . I opened the file and read all the lines, stripping any extra spaces or
newline characters. This gave me a list of tweets to work with.
3.Step 3: Access Token for Hugging Face : I included an access token for Hugging Face, which is
necessary if the model I want to use requires authentication. This token allows me to access the
model.
4.Step 4: Initialize the Tokenizer and Model : I then initialized the tokenizer and the model from
Hugging Face. The model I chose is "meta llama/Llama 2 13b hf". I used the access token here to
ensure I could load the model properly. The tokenizer is responsible for converting text into a
format that the model can understand.
5.Step 5: Add a Padding Token if Necessary : I checked if the tokenizer had a padding token. If it
didn't, I set the padding token to be the same as the end of sequence token. Padding tokens are
important to make sure all input sequences are of the same length
6.Step 6: Load the Model I loaded the LLaMA2 13B model and set it to evaluation mode. Evaluation mode is
used when I'm only generating outputs or embeddings and not training the model. I also moved the model
to the GPU (if available) to speed up the computations.
7. Step 7: Define a Function to Generate Embeddings I defined a function generate_embeddings_batched to
generate embeddings for the tweets. This function processes the tweets in batches, which helps in
managing memory and speeding up the process. For each batch, I tokenized the text, ran it through the
model, and extracted the hidden states from the last layer of the model. I then took the average of these
hidden states to get the embeddings for each tweet.
8.Step 8: Generate Embeddings for Tweets I used the function to generate embeddings for all the tweets. By
processing the tweets in batches of 8, I managed the memory usage and ensured the process was efficient.
9.Step 9: Save the Embeddings to a CSV File Finally , I saved the generated embeddings to a CSV file named
week6_embeddings_llama2 13b_batched3.csv . This allowed me to have a structured format of the
embeddings , which I can later use for analysis or other tasks




5_cosine_similarity-
1) The next step according to the research paper is querying the dataset and finding out how many relevant
tweets can it extract First we manually tell
gpt to generate a tweet Just on the CIF simply giving it no other context. Most of the
tweets didn't even have any mention of flood or disaster. Now we use that one single tweet to generate just
one embedding using the llama2 13b model . After That we use our large embedding file and apply cosine
similarity on that file to find out the top 150 most relevant tweets for our discussion
2)In The second step we ask
gpt to generate a tweet an this time the prompt contains both CIF+ Impact label.
We also tell it to generate it in regard to the kerala floods 2018 . We again repeat step 1 and this time we gather
60 most relevant tweets out of those top 150 tweets using cosine similarity
3)We again repeat step one This time generating a tweet which contains CIF + “disaster impacts” and we get
the top 10 tweets using cosine similarity
Note: In totality there are 2 python scripts the does the job
The first one generate embedding for all the three tweets and store them in separate csv files
All three tweets are initialized as string in the code and we use LLAMA
2 13B to generate embedding's for
them and store them into separate csv file
The second python script uses python libraries to obtain cosine similarities and store the respective 10 most
relevant tweets inside a directory in a text file
What I concluded that even after taking the most relevant and high quality tweets out of the top 10 tweets only
4-5 tweets were actually about the CIF rest all were not very relevant and were not even about the CIF. For instance, the chosen CIF wasVengola Government Homeo Dispensary. Ifollowed all the steps mentioned previously, and you can clearly see that only 5tweets were actually about the chosen CIF. This is not an isolated case. I repeated this process for 3 4 more CIFs, and the accuracy was even lower. In those cases, only 1 2 tweets were actually relevant.


6_operational_status.py-
The next step was to classify the tweets based on operational status, severity, and
impact. The research paper used relevant tweets and employed the Mistral 7B v1.0
model for classification. However, since v1.0 was not publicly available, I attempted to
use v0.3 and v0.1. Both versions required fine tuning for our dataset, but unfortunately,
my code didn't classify the tweets correctly As a last resort, I used the next best model for this task, which was the LLaMA 2 13B
model. I combined the model's results with string comparison techniques to predict the
operational status, severity, and impact label
The significance of the softmax probability and threshold value in this code is crucial
for determining the operational status of a tweet. The softmax probability converts the
model's raw output into a set of probabilities that sum to one, indicating the likelihood
of each possible status. The threshold value handles uncertainty: if the highest softmax
probability (indicating the model's confidence in its prediction) is below this threshold,
the operational status is classified as 'unknown'. This ensures that only sufficiently
confident predictions are accepted, improving the overall reliability of the classification.
Additionally, we used string comparison to identify specific keywords in the tweets,
ensuring accurate labeling.



7_quality_scores

![image](https://github.com/user-attachments/assets/8f7cfb75-6363-4f8f-aa61-ba7482fffc6b)



8_plot

![image](https://github.com/user-attachments/assets/5dd35aff-350a-41b5-9cab-2131c6b7bdef)




