from flask import Flask, render_template, url_for, request
import numpy as np

import random

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home(title=None):
    title="Home"
    return render_template("home.html", title=title)

@app.route('/about')
def about():
    return render_template("about.html")



if __name__=="__main__":
    app.run(debug=True)
