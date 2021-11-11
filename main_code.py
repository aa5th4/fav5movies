#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")


# In[ ]:


movies.merge(credits,on='title')


# In[ ]:


movies=movies.merge(credits,on='title')


# In[ ]:


movies.info()


# In[ ]:


movies.keys()


# In[ ]:


movies=movies[['movie_id', 'cast', 'crew', 'overview','keywords','genres','title']]


# In[ ]:


movies.isnull().sum()


# In[ ]:


movies.dropna()


# In[ ]:


movies.iloc[0].genres


# we want only name values line Action, fantasy from genres so we have to clean it

# In[ ]:


import ast
def convert(obj):
    list=[]
    for i in ast.literal_eval(obj):
        list.append(i['name'])
    return list


# ast lib. is to convert string into list .first we have to convert genre data into list then we can extract name values from there 

# In[ ]:


movies['genres']= movies['genres'].apply(convert)


# In[ ]:


movies.head()


# In[ ]:


movies['keywords']=movies['keywords'].apply(convert)


# In[ ]:


def convert2(obj):
    list=[]
    a=0
    for i in ast.literal_eval(obj):
        if a != 3:
            list.append(i['name'])
            a=a+1
        else:
            break
    return list


# In[ ]:


movies['cast']=movies['cast'].apply(convert2)


# In[ ]:


movies.head()


# In[ ]:


movies['cast'][0]


# In[ ]:


def convert3(obj):
    list=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            list.append(i['name'])
            break
    return list


# In[ ]:


movies['crew']=movies['crew'].apply(convert3)


# In[ ]:


movies['crew'][0]


# In[ ]:


movies['overview'][0]


# In[ ]:


movies['overview']= movies['overview'].apply(lambda x:str(x).split())


# In[ ]:


movies['overview']


# In[ ]:


movies['genres']= movies['genres'].apply( lambda x:[i.replace(" ","") for i in x])


# we have to remove space between names so that our model dont get confused by same first_names

# In[ ]:


movies['cast']= movies['cast'].apply( lambda x:[i.replace(" ","") for i in x])
movies['crew']= movies['crew'].apply( lambda x:[i.replace(" ","") for i in x])
movies['keywords']= movies['keywords'].apply( lambda x:[i.replace(" ","") for i in x])


# In[ ]:


movies.head()


# now combine these columns and put them in new column called tags

# In[ ]:


movies['tags'] =movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[ ]:


new_df=movies[['movie_id','title','tags']]


# join method join list item 

# In[ ]:


new_df.head()


# In[ ]:


new_df['tags'][0]


# In[ ]:


new_df


# In[ ]:


new_df['tags']= new_df['tags'].apply(lambda x:" ".join(x))


# In[ ]:


new_df.head()


# In[ ]:


new_df['tags'][0]


# In[ ]:


new_df['tags'] =new_df['tags'].apply(lambda x:x.lower())


# In[ ]:


new_df.head()


# In[ ]:


import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[ ]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[ ]:


new_df['tags']=new_df['tags'].apply(stem)


# In[ ]:


new_df['tags'][0]


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors= cv.fit_transform(new_df['tags']).toarray() 


# In[ ]:


cv.get_feature_names()


# In[ ]:


vectors


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity 
similarity = cosine_similarity(vectors)


# cosine similarity calculate the cosine distance of each movie with every movie and then after getting distance we want 5 movies which is most similar to that given movie (5 movies whose cosine distance is more) for that we sorted similarity in descending order without losing index position with help of enumerate function  

# In[ ]:


similarity[0]


# In[ ]:


sorted(list(enumerate (similarity[0])),reverse =True,key = lambda x:x[1])


# In[ ]:


sorted(list(enumerate (similarity[0])),reverse =True,key = lambda x:x[1])[1:6]


# In[ ]:


def recommand(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movie_dist=sorted(list(enumerate (distances)),reverse =True,key = lambda x:x[1])[1:6]
    for i in movie_dist:
        print(new_df.iloc[i[0]].title)


# In[ ]:


recommand('Avatar')


# In[ ]:


import pickle 
pickle.dump(new_df.to_dict(),open('movie_dist.pkl','wb'))


# In[ ]:



pickle.dump(similarity,open('similar.pkl','wb'))


# In[ ]:




