
import boto3
import pandas as pd
import numpy as np
import sys
from io import StringIO
import surprise
from surprise import accuracy
from sklearn.model_selection import train_test_split
from surprise import SVD, NMF, KNNBaseline
import pickle
from flask import Flask, request
from flask import render_template
import pickle
import pandas as pd
from io import BytesIO

# Initialize app
app = Flask(__name__)

# load the pickled models and dataframes

with open('../../pickle_recom/recom_colab.pkl', 'rb') as f:
    recom_colab = pickle.load(f)

with open('../../pickle_recom/recom_content.pkl', 'rb') as f:
    recom_content = pickle.load(f)

with open('../../pickle_recom/df_items.pkl', 'rb') as f:
    df_items = pickle.load(f)

with open('../../pickle_recom/df_reviewer.pkl', 'rb') as f:
    df_reviewer = pickle.load(f)



# Home page with form on it to submit new data
@app.route('/')
def get_new_data():
    return '''
    <body>  
    	<style>
        .center {
        display: block;
        margin-left: auto;
        margin-right: auto;
        }
		h1 {text-align: center;}
		h2 {text-align: center;}
		p {text-align: center;}
        body {background-image: url('https://images.squarespace-cdn.com/content/v1/55cd5e6de4b0af9801dd7aa7/1587662791718-G9371DYEL9H02TPQFBNU/ke17ZwdGBToddI8pDm48kL4WrIntsHuCODFzGytxs8sUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYxCRW4BPu10St3TBAUQYVKcw31z2cKmL83lZVTgYf1Shcnt0pzT4b-h8WwoQ3rX-86z0Q_QpJgDA4jmv5AtYw-J/SoHoViews.jpg?format=1500w');
                background-size: cover;
                background-repeat: no-repeat}
		form {text-align: center;}
		img {text-align: center;}
	</style>
        <div>
        <h1 style="color:White; background-color:Navy;">Amazon Products Recommender</h1>
        <h2>Sports & Outdoors</h2>
        <p>For new users, enter the search term and for current users enter User ID</p>
        <div id="banner" style="overflow: hidden; display: flex; justify-content:space-around;">
        <form action="/predict-new" method='POST'>
          Product Name:<br>
          <input type="text" placeholder="Search.." name="searchterm"> 
          <br><br>
          <input type="submit" value="New User">
        </form>
        <form action="/predict-current" method='POST'>
          User ID:<br>
          <input type="text" placeholder="User ID.." name="userID"> 
          <br><br>
          <input type="submit" value="Current User">
        </form>
        </div>
        </div>
        </body>
    '''

@app.route('/predict-current', methods = ["GET", "POST"])
def predict2():

    recommended_items = recom_content.get_recommendations(item)

    userID = request.form['userID']
    #recomms = pkl_func2(userID)
    recomms = pd.DataFrame({"asin":[1, 2, 3, 4, 5],
                "title":["A", "B", "C", "D", "E"],
                "links":["https://images-na.ssl-images-amazon.com/images/I/717Y104so7L._SL1500_.jpg",
                "https://images-na.ssl-images-amazon.com/images/I/510ygOpgnAL._SY550_.jpg",
                "https://images-na.ssl-images-amazon.com/images/I/31i3sjXGebL._SX425_.jpg",
                "https://images-na.ssl-images-amazon.com/images/I/51YTdPxBU4L._SY355_.jpg",
                "https://images-na.ssl-images-amazon.com/images/I/91r7p2GP0UL._AC_SX425_.jpg"]})

    page = f'''
        <form action="/predict-new" method='POST'>
          Product Name:<br>
          <input type="text" placeholder="Search.." name="searchterm"> 
          <br><br>
          <input type="submit" value="Search">
        </form>
        <h2>You might also like</h2>
        <div id="banner" style="overflow: hidden; display: flex; justify-content:space-around;">
        <p margin-top: 10;>{recomms.loc[0, "title"]}</p>
        <img id="product_img1" src={recomms.loc[0, "links"]} alt="product 1"  style = "display: inline-block; height:200px;">
        <p margin-top: 10em;>{recomms.loc[1, "title"]}</p>
        <img id="product_img2" src={recomms.loc[1, "links"]} alt="product 2" style = "display: inline-block; height:200px;">
        <p margin-top: 10em;>{recomms.loc[2, "title"]}</p>
        <img id="product_img3" src={recomms.loc[2, "links"]} alt="product 3" style = "display: inline-block; height:200px;">
        <p margin-top: 10em;>{recomms.loc[3, "title"]}</p>
        <img id="product_img4" src={recomms.loc[3, "links"]} alt="product 4" style = "display: inline-block; height:200px;">
        <p margin-top: 10em;>{recomms.loc[4, "title"]}</p>
        <img id="product_img5" src={recomms.loc[4, "links"]} alt="product 5" style = "display: inline-block; height:200px;">
        </div>
    '''
    return page

@app.route('/predict-new', methods = ["GET", "POST"])
def predict():

    
    searchterm = request.form['searchterm']
    # request the text from the form 
    # df = pd.DataFrame({"asin": ["A1", "B2"],
    #           "reviwerID": ["001", "007"],
    #           "title": ["door handle Sonata", "Rimmel lipstick"],
    #           "recomms": [["s1", "s2", "s3", "s4", "s5"], ["l1", "l2", "l3", "l4", "l5"]],
    #           "image_x": ["https://images.app.goo.gl/kd8gyDu6fnGNGuwW8", "https://images-na.ssl-images-amazon.com/images/I/717Y104so7L._SL1500_.jpg"]})

    
    df_items["condition"] = df_items["title"].apply(lambda x: searchterm in x)

    this_asin = df_items[df_items["condition"]==True].index[0]
    #add codition for empty this_asin
    item = this_asin
    recommended_items = recom_content.get_recommendations(item)
    recomms = df_items.loc[recommended_items]
    # recomms = pd.DataFrame({"asin":[1, 2, 3, 4, 5],
    #             "title":["A", "B", "C", "D", "E"],
    #             "links":["https://images-na.ssl-images-amazon.com/images/I/717Y104so7L._SL1500_.jpg",
    #             "https://images-na.ssl-images-amazon.com/images/I/510ygOpgnAL._SY550_.jpg",
    #             "https://images-na.ssl-images-amazon.com/images/I/31i3sjXGebL._SX425_.jpg",
    #             "https://images-na.ssl-images-amazon.com/images/I/51YTdPxBU4L._SY355_.jpg",
    #             "https://images-na.ssl-images-amazon.com/images/I/91r7p2GP0UL._AC_SX425_.jpg"]})

    page = f'''
        <h2>The selected item</h2>
        <p margin-top: 10;>{df_items.loc[this_asin]["title"]}</p>
        <img id="product_img1" src={df_items.loc[this_asin]["links"][-1]} alt="product 1"  style = "display: inline-block; height:100px;">
        <h2>Similar items:</h2>
        <p margin-top: 10;>{recomms["title"][0]}</p>
        <img id="product_img1" src={recomms["links"][0][-1]} alt="product 1"  style = "display: inline-block; height:100px;">
        <p margin-top: 10em;>{recomms["title"][1]}</p>
        <img id="product_img2" src={recomms["links"][1][-1]} alt="product 2" style = "display: inline-block; height:100px;">
        <p margin-top: 10em;>{recomms["title"][2]}</p>
        <img id="product_img3" src={recomms["links"][2][-1]} alt="product 3" style = "display: inline-block; height:100px;">
        <p margin-top: 10em;>{recomms["title"][3]}</p>
        <img id="product_img4" src={recomms["links"][3][-1]} alt="product 4" style = "display: inline-block; height:100px;">
        <p margin-top: 10em;>{recomms["title"][4]}</p>
        <img id="product_img5" src={recomms["links"][4][-1]} alt="product 5" style = "display: inline-block; height:100px;">
        
    '''
    return page

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
