import praw
import sqlite3

conn = sqlite3.connect('reddit.db')
c = conn.cursor()
c.execute('''DROP TABLE IF EXISTS topics ''')
c.execute('''DROP TABLE IF EXISTS comments''')
c.execute('''CREATE TABLE topics(topicTitle text, topicText text, topicID text, topicCategory text)''')
c.execute(
    '''CREATE TABLE comments(commentText text, commentID text, topicTitle text, topicText text, topicID text, topicCategory text)''')
userAgent = 'user'
clientID = 'V9smqLUZ8hti6w'  #
clientSecret = 'oFXA2j-GjiVEEGVTiaVCfeSK-1Q'  # REGISTER on reddit.com and acquire this fields
r = praw.Reddit(user_agent=userAgent, client_id=clientID, client_secret=clientSecret)
subreddits = ['bioinformatics', 'datascience']
my_limit = 100


###################################################################
def prawGetData(mlimit, subredditName):
    topics = r.subreddit(subredditName).hot(limit=mlimit)
    commentInsert = []
    topicInsert = []
    topicNBR = 1
    # '  Title: '+str(topic.title)+'
    for topic in topics:
        nn = (float(topicNBR) / mlimit) * 100
        if nn in range(100):
            print('******** TOPIC_ID: ' + str(topic.id) + '******* COMPLETE: ' + str(nn) + ' % ***')

        topicNBR += 1
        try:
            topicInsert.append((topic.title, topic.selftext, topic.id, subredditName))
        except:
            pass
        try:
            for comment in topics.comment:
                commentInsert.append(comment.body, comment.id, topic.title, topic.selftext, topic.id, subredditName)
        except:
            pass

    print('**************************')
    print('INSERTING DATA INTO SQLITE')
    c.executemany('INSERT INTO topics VALUES (?,?,?,?)', topicInsert)
    print('INSERTED TOPICS')
    c.executemany('INSERT INTO comments VALUES (?,?,?,?,?,?)', commentInsert)
    print('INSERTED COMMENTS')
    conn.commit()


###################################################################


for subject in subreddits:
    prawGetData(mlimit=my_limit, subredditName=subject)
