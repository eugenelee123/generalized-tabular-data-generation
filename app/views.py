from flask import flash, redirect, request, render_template, Flask, make_response
from app import app, DataAugmentation
import requests
import json
import pandas as pd


@app.route('/', methods = ['GET','POST'])
def data_aug():

    if request.method == 'GET':
        return render_template('home.html')

    elif request.method == 'POST':
        # if 'file' not in request.files:
        #     return render_template('home.html', msg='No file selected')
        
        discrete_columns = request.form['discrete_columns']
        target_name = request.form['target_name']
        request_file = request.files['file']
        dataframe = pd.read_csv(request_file)

        discrete_columns = discrete_columns.split(",")
        print(discrete_columns)
        
        optimalSamplesCount = DataAugmentation.sgd(dataframe, target_name, discrete_columns)
        syntheticData = DataAugmentation.generateOptimalDataSamples(dataframe,optimalSamplesCount)
        
        resp = make_response(syntheticData.to_csv())
        resp.headers['Content-Disposition'] = 'attachment; filename = syntheticData.csv'
        resp.headers['Content-Type'] = 'text/csv'
        return resp
        
        # return render_template('home.html')