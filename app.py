import streamlit as st
import mysql.connector
import pandas as pd
import requests
import os
import ast
import time
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

# Database Setup
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="movie_app"
    )

def authenticate(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username=%s AND password=%s"
    cursor.execute(query, (username, password))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def signup_user(username, password, email):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password, email) VALUES (%s, %s, %s)", (username, password, email))
        conn.commit()
        return True
    except mysql.connector.IntegrityError:
        return False
    finally:
        conn.close()


# Styling
def set_background():
    st.markdown(
        """
        <style>
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            height: 100%;
            width: 100%;
            background-image: url("https://img.freepik.com/free-vector/realistic-film-roll_1017-6356.jpg");
            background-size: cover;
            background-position: center;
            filter: blur(8px);
            z-index: -1;
        }
        .stApp {
            background: transparent;
        }

        """,
        unsafe_allow_html=True
    )


def set_full_page_height():
    st.markdown("""
        <style>
        .main {
            min-height: 100vh;  /* Full viewport height */
        }
        </style>
    """, unsafe_allow_html=True)




def extract_names(text, max_items=None):
    try:
        items = ast.literal_eval(text)
        names = [item['name'].replace(" ", "") for item in items]
        return names[:max_items] if max_items else names
    except:
        return []

def extract_director(text):
    try:
        items = ast.literal_eval(text)
        for item in items:
            if item['job'] == 'Director':
                return [item['name'].replace(" ", "")]
        return []
    except:
        return []


# data loading
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv', low_memory=False)
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].dropna()

    def extract_names(text, max_items=None):
        try:
            items = ast.literal_eval(text)
            names = [item['name'].replace(" ", "") for item in items]
            return names[:max_items] if max_items else names
        except:
            return []

    def extract_director(text):
        try:
            items = ast.literal_eval(text)
            for item in items:
                if item['job'] == 'Director':
                    return [item['name'].replace(" ", "")]
            return []
        except:
            return []

    movies['genres'] = movies['genres'].apply(extract_names)
    movies['keywords'] = movies['keywords'].apply(extract_names)
    movies['cast'] = movies['cast'].apply(lambda x: extract_names(x, max_items=3))
    movies['crew'] = movies['crew'].apply(extract_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
    final_df = movies[['movie_id', 'title', 'tags']]
    return movies, final_df



@st.cache_data
def get_details(movie_id):
    api_key = "8265bd1679663a7ea12ac168da84d2e8"

    movie_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    movie_data = requests.get(movie_url).json()

    credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={api_key}&language=en-US"
    credits_data = requests.get(credits_url).json()

    title = movie_data.get('title', 'N/A')
    release_year = movie_data.get('release_date', 'N/A')[:4] if movie_data.get('release_date') else 'N/A'
    runtime = f"{movie_data.get('runtime', 'N/A')} min"
    genres = [genre['name'].replace(" ", "") for genre in movie_data.get('genres', [])]
    vote_average = movie_data.get('vote_average', 'N/A')
    country = movie_data.get('production_countries', [])
    countries = [c['name'].replace(" ", "") for c in country]

    director = ''
    for crew in credits_data.get('crew', []):
        if crew['job'] == 'Director':
            director = crew['name'].replace(" ", "")
            break

    cast = credits_data.get('cast', [])
    top_cast = [actor['name'].replace(" ", "") for actor in cast[:3]]

    tag = (
        f"**Name:** \n{title}\n"
        f"**Release year:** \n{release_year}\n "
        f"**Genres:** \n{' '.join(genres)}\n"
        f"**Runtime:** \n{runtime}\n"
        f"**Top cast:** \n{' '.join(top_cast)}\n"
        f"**Avg rating:** \n{vote_average}\n"
        f"**Director:** \n{director}\n"
        f"**Country:** \n{' '.join(countries)}\n"
    )
    return tag


@st.cache_data
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path', '')
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

# content based recomandation

def recommend(movie_title):
    movies, final_df = load_data()
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(final_df['tags']).toarray()
    model = NearestNeighbors(n_neighbors=6, metric='cosine')
    model.fit(vector)

    if movie_title not in final_df['title'].values:
        return [], []
    index = final_df[final_df['title'] == movie_title].index[0]
    distances, indices = model.kneighbors([vector[index]])
    movie_names = []
    movie_posters = []
    for i in indices[0][1:]:
        movie_id = movies.iloc[i].movie_id
        movie_names.append(final_df.iloc[i].title)
        movie_posters.append(fetch_poster(movie_id))
    return movie_names, movie_posters

def profile_dropdown():
    with st.sidebar:
        st.markdown("## 👤 Profile")
        st.markdown(f"**User:** {st.session_state['current_user']}")
        if st.button("🏠 Home"):
            st.session_state['page'] = 'home'
            st.rerun()
        # if st.button("📜 History"):
        #     st.success("Showing your watch history... (Coming soon)")
        if st.button("🚪 Logout"):
            st.session_state["logged_in"] = False
            st.rerun()

def get_user_id(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT id FROM users WHERE username = %s"
    cursor.execute(query, (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]  # Returns the 'id' value
    return None

def save_rating(username, movie_id, rating):
    user_id = get_user_id(username)
    movie = pd.read_csv('tmdb_5000_movies.csv')
    movie_id = int(movie_id)  # Ensure it's an int
    movie['id'] = movie['id'].astype(int)

    if user_id is None:
        print(f"User '{username}' not found in the database.")
        return False

    filename = 'new_rating.csv'
    new_entry = {'userId': user_id, 'movieId': int(movie_id), 'rating': float(rating)}

    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=['userId', 'movieId', 'rating'])

    if not df.empty:
        df['userId'] = df['userId'].astype(int)
        df['movieId'] = df['movieId'].astype(int)

    try:
        title = movie.loc[movie['id'] == movie_id, 'title'].values[0]
    except IndexError:
        title = "Unknown"

    match_index = ((df['userId'] == new_entry['userId']) & (df['movieId'] == new_entry['movieId']))

    info_box = st.empty()  # Create a placeholder
    if match_index.any():
        existing_rating = df.loc[match_index, 'rating'].values[0]
        df.loc[match_index, 'rating'] = new_entry['rating']
        info_box.info(f"Previous rating was: {existing_rating} \nUpdated '{title}' with {rating} ⭐")
    else:
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        info_box.info(f"Rated '{title}' with {rating} ⭐")

    time.sleep(3)
    info_box.empty()

    df.to_csv(filename, index=False)
    return True


# with st.form(key=f"update_form_{user_id}_{movie_id}"):
#     choice = st.radio("Do you want to update the rating?", ["Yes", "No"])
#     submitted = st.form_submit_button("Submit")
#
#     if submitted:
#         if choice == "Yes":
#             df.loc[match_index, 'rating'] = new_entry['rating']
#             st.success(f"Rated '{title}' with {rating} ⭐ (Rating updated.)")
#         else:
#             st.info("Rating not changed.")
#


# c= st.radio(f"Do you want to update your rating for this movie?",
#                             ["-- Select --", "No", "Yes"],
#                             key=f"{user_id}_{movie_id}"
#                         )
# c=input("fRating already exists for {title} movie do you want to update it?")


# authentication Page
def auth_page():
    set_background()
    set_full_page_height()
    st.markdown("<h1 style='text-align: center; color: black;'>🎬 Movie Recommendation System</h1>",
                unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])

    with tab1:
        st.text_input("Username", key="login_user")
        st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if authenticate(st.session_state.login_user, st.session_state.login_pass):
                st.session_state["logged_in"] = True
                st.session_state["current_user"] = st.session_state.login_user
                st.session_state["user_id"] = get_user_id(st.session_state.login_user)
                st.session_state['page'] = 'home'
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        st.text_input("New Username", key="signup_user")
        st.text_input("New Password", type="password", key="signup_pass")
        st.text_input("Email", key="signup_mail")

        if st.button("Sign Up"):
            username = st.session_state.signup_user.strip()
            password = st.session_state.signup_pass.strip()
            email = st.session_state.signup_mail.strip()

            # Basic validations
            if not username or not password or not email:
                st.error("All fields are required.")
            elif "@" not in email or not email.endswith(".com"):
                st.error("Enter a valid email address")
            else:
                if signup_user(username, password, email):
                    st.success("Account created! You can now log in.")
                else:
                    st.warning("Username already exists. Try another.")


#Main Interface
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    auth_page()
else:
    set_background()
    profile_dropdown()
    d,movies=load_data()

    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'

    if st.session_state['page'] == 'home':
        st.title(f"🍿 Welcome {st.session_state['current_user']} to the Movie Recommendation System")
        st.write("Explore personalized recommendations below!")
        movie = pd.read_csv('tmdb_5000_movies.csv')

        st.subheader("🔍 Search for a Movie")
        if "clear_search" not in st.session_state:
            st.session_state.clear_search = False

        if st.session_state.clear_search:
            st.session_state.search_query = ""
            st.session_state.clear_search = False

        search_query = st.text_input("Enter movie name", key="search_query")

        if search_query.strip():
            matched_movies = movie[movie['title'].str.contains(search_query.strip(), case=False, na=False)]
            if not matched_movies.empty:
                for idx, row in matched_movies.iterrows():
                    movie_id = row['id']
                    poster_url = fetch_poster(movie_id)

                    st.image(poster_url, width=150)
                    st.markdown(f"### {row['title']}")

                    rate_key = f"show_rate_{movie_id}"
                    detail_key = f"show_detail_{movie_id}"

                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("Rating", key=rate_key):
                            st.session_state[f"selected_view_{movie_id}"] = "rating"
                    with b2:
                        if st.button("Details", key=detail_key):
                            st.session_state[f"selected_view_{movie_id}"] = "details"

                    selected_view = st.session_state.get(f"selected_view_{movie_id}", None)
                    info_box = st.empty()
                    if selected_view == "rating":
                        rating = st.slider("Rate this movie", 1.0, 5.0, step=0.1,
                                           label_visibility="collapsed", key=f"rate_{movie_id}")

                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("Submit", key=f"submit_{movie_id}"):
                                st.session_state[f"select_view_{movie_id}"] = "submit"
                                # save_rating(st.session_state['current_user'], movie_ids[idx], rating)

                        with c2:
                            if st.button("Close", key=f"close_btn_{movie_id}"):
                                st.session_state[f"select_view_{movie_id}"] = "close"

                        select_view = st.session_state.get(f"select_view_{movie_id}", None)
                        if select_view == "submit":
                            save_rating(st.session_state['current_user'], movie_id, rating)

                        elif select_view == "close":
                            selected_key = f"selected_view_{movie_id}"
                            action_key = f"select_view_{movie_id}"
                            for key in [selected_key, action_key]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.rerun()

                    elif selected_view == "details":
                        info_box.info(f"{get_details(movie_id)}")
                        if st.button("clear", key=f"clear_btn_{movie_id}"):
                            selected_key = f"selected_view_{movie_id}"
                            if selected_key in st.session_state:
                                del st.session_state[selected_key]
                            st.rerun()

                if st.button("Close Search"):
                    st.session_state.clear_search = True
                    st.rerun()

            else:
                st.warning("Movie not found.")


        st.markdown("---")
        st.subheader("✨ Get Recommendations")
        selected_movie = st.selectbox("Select a movie", movies['title'].values)

        if st.button("Show Recommendation"):
            recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
            st.markdown("### Recommended Movies")
            rec_cols = st.columns(5)
            for i in range(5):
                with rec_cols[i]:
                    st.image(recommended_movie_posters[i])
                    st.text(recommended_movie_names[i])
            if st.button("Back"):
                st.session_state['page'] = 'home'


        # user-Based Recommendations
        st.markdown("---")
        st.subheader("👥 Recommended for You (User-Based)")

        @st.cache_data
        def load_data():
            movie_df = pd.read_csv('tmdb_5000_movies.csv')
            ratings_df = pd.read_csv('new_rating.csv')
            return movie_df, ratings_df

        movie_df, ratings_df = load_data()

        ratings_df['userId'] = ratings_df['userId'].astype(int)
        ratings_df['movieId'] = ratings_df['movieId'].astype(int)

        user_id = st.session_state.get('user_id')

        if user_id is not None:
            movie_id_to_title = dict(zip(movie_df['id'], movie_df['original_title']))

            # Create user-item matrix
            user_movie_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

            if user_id in user_movie_matrix.index:
                user_ratings_count = (user_movie_matrix.loc[user_id] > 0).sum()
                if user_ratings_count < 5:
                    st.info("You need to rate at least 5 movies to get personalized suggestions.")
                else:
                    sparse_matrix = csr_matrix(user_movie_matrix.values)
                    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
                    knn_model.fit(sparse_matrix)

                    user_index = user_movie_matrix.index.tolist().index(user_id)

                    distances, indices = knn_model.kneighbors([user_movie_matrix.iloc[user_index]], n_neighbors=6)
                    similar_users = user_movie_matrix.iloc[indices.flatten()[1:]]

                    mean_ratings = similar_users.mean().sort_values(ascending=False)

                    user_rated_movies = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index
                    recommendations = mean_ratings.drop(user_rated_movies, errors='ignore')

                    top_n = 5
                    st.subheader("📌 Top Movie Recommendations for You:")
                    cols = st.columns(top_n)

                    for i, movie_id in enumerate(recommendations.index[:top_n]):
                        movie_title = movie_id_to_title.get(movie_id, "Unknown Title")
                        poster_url = fetch_poster(movie_id)

                        with cols[i]:
                            st.image(poster_url, width=150)
                            st.markdown(f"**{movie_title}**")

            else:
                st.warning("You have to rerun the program for personalized suggestions.")
        else:
            st.warning("User not logged in or user_id not set.")

        st.markdown("---")
        st.subheader("🎥 All Movies")

        movies_per_page = 20
        total_movies = len(movie)
        total_pages = (total_movies + movies_per_page - 1) // movies_per_page

        if "current_page" not in st.session_state:
            st.session_state.current_page = 1

        start_idx = (st.session_state.current_page - 1) * movies_per_page
        end_idx = min(start_idx + movies_per_page, total_movies)

        all_movie_titles = movie['title'].values[start_idx:end_idx]
        movie_ids = movie['id'].values[start_idx:end_idx]

        container = st.container()
        scroll_cols = container.columns(4)

        for idx in range(len(all_movie_titles)):
            col = scroll_cols[idx % 4]
            with col:
                poster = fetch_poster(movie_ids[idx])
                st.image(poster, width=120)

                title = all_movie_titles[idx]
                st.markdown(
                    f"""
                    <div style="height: 72px; overflow: hidden; text-align: center;">
                        <p style="margin:0">{title}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                rate_key = f"show_rate_{movie_ids[idx]}"
                detail_key = f"show_detail_{movie_ids[idx]}"

                b1, b2 = st.columns(2)
                with b1:
                    if st.button("Rating", key=rate_key):
                        st.session_state[f"selected_view_{movie_ids[idx]}"] = "rating"
                with b2:
                    if st.button("Details", key=detail_key):
                        st.session_state[f"selected_view_{movie_ids[idx]}"] = "details"

                selected_view = st.session_state.get(f"selected_view_{movie_ids[idx]}", None)
                info_box = st.empty()
                if selected_view == "rating":
                    rating = st.slider("Rate this movie", 1.0, 5.0, step=0.1,
                                       label_visibility="collapsed", key=f"rate_{movie_ids[idx]}")

                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Submit", key=f"submit_{movie_ids[idx]}"):
                            st.session_state[f"select_view_{movie_ids[idx]}"] = "submit"
                            # save_rating(st.session_state['current_user'], movie_ids[idx], rating)

                    with c2:
                        if st.button("Close", key=f"close_btn_{movie_ids[idx]}"):
                            st.session_state[f"select_view_{movie_ids[idx]}"] = "close"

                    select_view = st.session_state.get(f"select_view_{movie_ids[idx]}", None)
                    if select_view == "submit":
                        save_rating(st.session_state['current_user'], movie_ids[idx], rating)

                    elif select_view == "close":
                        selected_key = f"selected_view_{movie_ids[idx]}"
                        action_key = f"select_view_{movie_ids[idx]}"
                        for key in [selected_key, action_key]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()

                elif selected_view == "details":
                    # You can customize this part to fetch and display movie details
                    info_box.info(f"{get_details(movie_ids[idx])}")
                    if st.button("clear", key=f"clear_btn_{movie_ids[idx]}"):
                        selected_key = f"selected_view_{movie_ids[idx]}"
                        if selected_key in st.session_state:
                            del st.session_state[selected_key]
                        st.rerun()

        st.markdown("---")
        new_page = st.number_input("Go to page", min_value=1, max_value=total_pages,
                                   value=st.session_state.current_page, step=1)

        # change page number
        if new_page != st.session_state.current_page:
            st.session_state.current_page = new_page
            st.rerun()