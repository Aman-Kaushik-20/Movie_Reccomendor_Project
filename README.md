# TMDB 5000 Movie Recommender Project

This project is aimed at building a movie recommendation system using the TMDB 5000 dataset. It involves preprocessing the data, extracting features, and using cosine similarity to recommend movies based on a given movie.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Usage](#usage)
5. [File Structure](#file-structure)
6. [Dependencies](#dependencies)

## Features

- **Data Loading**: Loading and merging TMDB 5000 movies and credits datasets.
- **Preprocessing**: Cleaning and preprocessing text data (genres, keywords, cast, crew, and overview).
- **Feature Extraction**: Using TF-IDF for feature extraction.
- **Similarity Calculation**: Using cosine similarity to find similar movies.
- **Movie Recommendation**: Recommending movies based on cosine similarity.

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/tmdb-movie-recommender.git
   cd tmdb-movie-recommender
   ```

2. **Install the required packages**:
   ```sh
   pip install -r requirements.txt
   ```

## Dataset

The dataset used for this project is the TMDB 5000 Movie Dataset. You can download it from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

Place the `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` files in the `data` directory.

## Usage

1. **Run the preprocessing and training script**:
   ```sh
   python main.py
   ```

2. **Access the results**:
   - The script will output the recommended movies for given movie titles.

## File Structure

```
tmdb-movie-recommender/
│
├── data/
│   ├── tmdb_5000_movies.csv        # TMDB 5000 movies dataset
│   ├── tmdb_5000_credits.csv       # TMDB 5000 credits dataset
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Preprocessing functions
│   ├── feature_extraction.py       # Feature extraction functions
│   ├── recommendation.py           # Recommendation functions
│
├── main.py                         # Main script to run the project
├── requirements.txt                # List of Python packages required
└── README.md                       # This README file
```

## Dependencies

- **numpy**: For numerical operations.
- **pandas**: For data manipulation.
- **nltk**: For natural language processing tasks.
- **scikit-learn**: For machine learning tasks.
- **ast**: For safely evaluating strings containing Python literals.
- **google-colab**: For mounting Google Drive in Colab.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Detailed Code Explanation

### Importing Required Libraries

```python
import numpy as np
import pandas as pd
import ast
from google.colab import drive
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
import pickle
```

### Loading the Dataset

```python
movies = pd.read_csv('data/tmdb_5000_movies.csv')
creds = pd.read_csv('data/tmdb_5000_credits.csv')

drive.mount('/content/drive')

movies = movies.merge(creds, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)
```

### Preprocessing Data

```python
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
```

### Feature Extraction

```python
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
features = tfidf.fit_transform(new_df['tags']).toarray()

ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

new_df['tags'] = new_df['tags'].apply(stem)
```

### Similarity Calculation

```python
similarity = cosine_similarity(features)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

recommend('Avatar')
recommend('Batman Begins')
```

### Saving the Model

```python
pickle.dump(new_df.to_dict(), open('movies.pl', 'wb'))
pickle.dump(similarity, open('similarity.pl', 'wb'))
```

This README file includes all the necessary details to understand, install, and run the project effectively.
