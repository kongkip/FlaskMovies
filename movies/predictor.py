from flask import Flask, render_template, request, url_for, redirect
import numpy as np
import catboost 
import json
from sklearn.externals import joblib
import json 
import pandas as pd

app = Flask(__name__)

@app.route("/")
def Home():

    return render_template("home.html")

@app.route("/predict", methods=["GET","POST"])
def predict():
    cat = catboost.CatBoostRegressor()
    budget_scaler = joblib.load("Pickled_Files/budget_scaler.pkl")
    model = cat.load_model("model")

    if request.method == "POST":
        votes = int(request.form["votes"])
        years = int(request.form["year"])
        actors = request.form["actors"]
        director = request.form["director"]
        genre = request.form["genre"]
        filminglocation = request.form["filminglocation"]
        writer = request.form["writer"]
        prodco= request.form["prodco"]
        rtime = int(request.form["movielength"])
        country = request.form["country"]
        budget = int(request.form["budget"])
        
        reshaped_budget = np.reshape(budget,(1,-1))
        new_budget = budget_scaler.transform(reshaped_budget)
        value = new_budget[0]
        
        data = [rtime,votes,genre,country,filminglocation,prodco,years,director,writer,actors,new_budget]
        df = pd.DataFrame(data=[data], columns=['rtime', 'votes', 'genre', 'country', 'filminglocation', 'prodco',
       'years', 'director', 'writer', 'actors', 'budget'])
        prediction = model.predict(df)
        print(writer)
        return json.dumps(prediction)
        '''  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
    <link rel="stylesheet" type="text/css" href="static/style.css">
    <script src="assets/code.js"></script>
    <link rel="stylesheet" type="text/css" href="static/style.css">
        <title>Predictions</title>
          <body>
      <h1>Predictions</h1>

      <h2><a style="color:white" href="http://127.0.0.1:5000/predict">Return To prediction Page</a></h2>
  </body>
        <h1 style="color:blue">Your rating is {}</h1>'''.format(prediction)
        

    return render_template("index1.html")
