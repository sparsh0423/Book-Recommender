from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import load_model

popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
similarity_scores = pickle.load(open('neural_similarity_scores.pkl', 'rb'))

app = Flask(__name__)
model = load_model('neural_collaborative_filtering_model.h5')


def neural_recommend(book_name):
    # Index fetch
    index = np.where(books['Book-Title'] == book_name)

    if index[0].size > 0:
        index = index[0][0]
    else:
        index = random.randint(0, 705)  # Changed from 700 to 705 since index size is 706
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        data.append(item)

    return data


@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )


@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')


@app.route('/recommend_books', methods=['POST'])
@app.route('/recommend_books', methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    index = np.where(pt.index == user_input)
    
    if index[0].size > 0:
        index = index[0][0]
    else:
        index = random.randint(0, len(pt.index) - 1)  # Select a random index within bounds
    
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        if i[0] < len(pt.index):  # Check if index is within bounds
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
            data.append(item)

    return render_template('recommend.html', data=data)



@app.route('/neural_recommend_books', methods=['POST'])
@app.route('/neural_recommend_books', methods=['POST'])
def neural_recommend_books():
    book_name = request.form.get('book_name')  # Extract book_name from form data
    neural_data = neural_recommend(book_name)  # Pass book_name to neural_recommend function
    return render_template('recommend.html', neural_data=neural_data)



if __name__ == '__main__':
    app.run(debug=True)
