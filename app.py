import flask
import pandas as pd

#reading the dataset
df=pd.read_csv('movies.csv')

#Storing movie titles from dataset
m_titles = [df['title'][i] for i in range(len(df['title']))]

#creating flask object
app = flask.Flask(__name__, template_folder='templates')


def create():
    from sklearn.feature_extraction.text import TfidfVectorizer

    #removing genral english words that occur
    t = TfidfVectorizer(stop_words='english')

    # nan value replaced by empty string
    df['overview'] = df['overview'].fillna('') 
    
    #generating the tfidf matrix
    matrix = t.fit_transform(df['overview'])
    
    return matrix


def calcosine():
    from sklearn.metrics.pairwise import cosine_similarity
    
    #cosine similarity applied
    cosine_sim = cosine_similarity(create(), create())
    
    return cosine_sim


def recommend(title, cosine_sim=calcosine()):
    #reverse mapping of movie title and dataframe indices
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    #getting index of inputed movie 
    index = indices[title]

   #finding cosine similarity score of inputed movie with other
    s = list(enumerate(cosine_sim[index]))

    #sorting  based on score
    s = sorted(s, key=lambda x: x[1], reverse=True)
    
    #taking  top 10 values
    s = s[1:11]
    
    movie_indices = [i[0] for i in s]
    
    return df['title'].iloc[movie_indices]


# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
    
    if flask.request.method == 'POST':
        m = flask.request.form['movie_name']
        
        if m not in m_titles:
            return (flask.render_template("wrong_input_result.html"))
        
        else:
            res = []
            names=recommend(m)
            for i in range(len(names)):
                res.append(names.iloc[i])
            return (flask.render_template("result.html",result=res,search_name=m))
            
	    

if __name__ == '__main__':
    app.run(debug=True)