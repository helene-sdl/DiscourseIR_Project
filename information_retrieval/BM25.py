import pandas as pd
from rank_bm25 import BM25Okapi
import re
import os


df = df = pd.read_csv("/Users/helene/DiscourseIR_Project/data/discourseIE_data.csv")


def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = text.split()
    return tokens

corpus = df["text"].tolist()
tokenized_corpus = [preprocess(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

def search(query, n=5):
    tokens = preprocess(query)
    scores = bm25.get_scores(tokens)    
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    results = df.iloc[top_n][["subreddit", "text"]].copy()
    results["score"] = [scores[i] for i in top_n]
    return results

queries = [
    #"feminists are destroying society",             
    #"men are victims of the system",                
    #"we need to protect women from violence",        
    #"statistics prove gender inequality",           
    #"I experienced discrimination personally",      
    #"feminism hurts men and boys",                   
    #"both sides have valid arguments",               
    #"the system is biased against fathers",          
    #"women deserve equal rights obviously",          
    #"toxic culture affects mental health",  
]
preprocessed_queries = [preprocess(q) for q in queries]
for q in queries:
    print(f"Query: {q}")
    results = search(q)
    for i, row in results.iterrows():
        print(f"\n[{row['subreddit']}] (score: {row['score']:.2f})")
        print(row["text"][:200] + "...")

#-->these are for testing the BM25, but i would like to do it the other way around actually: examples of us-vs-them framing, victim framing, 
#+
#what topics are talked about, what are the most common themes, what are the most common words?
#most common strategies used in the discourse, how do they differ across subreddits, how do they differ across time

#is this then a classification task? can we train a model to classify the strategies used in the discourse? can we train a model to classify the themes of the discourse? 
#could this further be used for hate speech detection? sexist language? --> different project tho 

#These are the questions I actually want to answer... but i don't think these can be answered with BM25

