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
    #"workplace harassment",
    #"gender pay gap",
    "domestic violence",
    #"men's mental health",
    #"parental rights custody"
]

for q in queries:
    print(f"Query: {q}")
    results = search(q)
    for i, row in results.iterrows():
        print(f"\n[{row['subreddit']}] (score: {row['score']:.2f})")
        print(row["text"][:200] + "...")

