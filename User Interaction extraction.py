import praw
import pandas as pd
reddit = praw.Reddit(client_id="", # your client id
client_secret="", # your client secret
user_agent="", # user agent name
username="", # your reddit username
password="")
subreddit = reddit.subreddit('depression')
posts = subreddit.hot(limit=200)
data = {'post_author': [], 'post': [], 'comment_author': [], 'comment': [], 'reply_author': [],'reply': []}
count = 0
for post in posts:
 if post.stickied or post.archived:
     continue
 if post.num_comments > 0:
     comments = post.comments.list()
     for comment in comments:
         if isinstance(comment, praw.models.MoreComments):
             continue
         replies = comment.replies.list()
         if len(replies) > 0:
             for reply in replies:
                 data['post_author'].append(post.author)
                 data['post'].append(post.selftext)
                 data['comment_author'].append(comment.author)
                 data['comment'].append(comment.body)
                 data['reply_author'].append(reply.author)
                 data['reply'].append(reply.body)
                 break # extract only one reply per comment
         else:
             continue
         break # extract only one comment per post
df4 = pd.DataFrame(data)
display(df4)
df4.to_csv("Depress.csv", index=False)
