import pandas as pd
fem_comments = pd.read_csv("https://raw.githubusercontent.com/AsmaeNakib/NLP-Project---Feminism-Discourse-on-Reddit/main/feminism_comments_as_posts.csv")

print(fem_comments.shape)
print(fem_comments.columns.tolist())
print(fem_comments["subreddit"].value_counts().head(10))

df = fem_comments[fem_comments["subreddit"].isin(["Feminism", "MensRights"])].copy()
df = df[df["text"].notna() & (df["text"].str.len() > 50)]  

print(df.shape)
print(df["subreddit"].value_counts())
print(df["text"].iloc[0])

df.to_csv("discourseIE_data.csv", index=False)
print(df[["subreddit", "text"]].head(3))