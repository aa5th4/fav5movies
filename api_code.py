import streamlit as st
import  pickle
import pandas as pd
import requests

def fetch_data(movie_id):
    response=requests.get('https://api.themoviedb.org/3/movie/{'
                          '}?api_key=a946ab47e8a979accec3df4ab65c11e6&language=en-US'.format(movie_id))
    data =response.json()
    return 'http://image.tmdb.org/t/p/w500/' +data['poster_path']



def recommand(movie):
    movie_index=movies[movies['title']==movie].index[0]
    distances=similarity[movie_index]
    movie_dist=sorted(list(enumerate (distances)),reverse =True,key = lambda x:x[1])[1:7]
    recommended_movies = []
    poster=[]
    for i in movie_dist:
        movie_id=movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        poster.append(fetch_data(movie_id))
    return recommended_movies,poster

st.title('my movies')

movie_dist=pickle.load(open('movie_dist.pkl','rb'))
similarity=pickle.load(open('similar.pkl','rb'))
movies=pd.DataFrame(movie_dist)

option = st.selectbox(
    'which movie you like to watch?',
    (movies['title'].values)
)

if st.button('Recommend'):
    names,posters = recommand(option)
    col1, col2, col3 ,col4,col5,col6 = st.columns(6)

    with col1:
        st.text(names[0])
        st.image(posters[0])

    with col2:
        st.text(names[1])
        st.image(posters[1])

    with col3:
        st.text(names[2])
        st.image(posters[2])

    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
    with col6:
        st.text(names[5])
        st.image(posters[5])
