from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def Home():
    return render_template('index.html')

@app.route('/cancer', methods=['GET', 'POST'])
def Cancer():
    return render_template('cancer.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def Diabetes():
    return render_template('diabetes.html')

@app.route('/heart', methods=['GET', 'POST'])
def Heart():
    return render_template('heart.html')

@app.route('/kidney', methods=['GET', 'POST'])
def Kidney():
    return render_template('kidney.html')

@app.route('/liver', methods=['GET', 'POST'])
def Liver():
    return render_template('liver.html')

@app.route('/malaria', methods=['GET', 'POST'])
def Malaria():
    return render_template('malaria.html')

@app.route('/pneumonia', methods=['GET', 'POST'])
def Pneumonia():
    return render_template('pneumonia.html')

@app.route('/cancer/predict', methods=['GET', 'POST'])
def Cancer_Predict():
    if request.method == 'POST':
        radius_mean = float(request.form['radius_mean'])
        texture_mean = float(request.form['texture_mean'])
        perimeter_mean = float(request.form['perimeter_mean'])
        area_mean = float(request.form['area_mean'])
        smoothness_mean = float(request.form['smoothness_mean'])
        compactness_mean = float(request.form['compactness_mean'])
        concavity_mean = float(request.form['concavity_mean'])
        concave_points_mean = float(request.form['concave_points_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
        radius_se = float(request.form['radius_se'])
        texture_se = float(request.form['texture_se'])
        perimeter_se = float(request.form['perimeter_se'])
        area_se = float(request.form['area_se'])
        smoothness_se = float(request.form['smoothness_se'])
        compactness_se = float(request.form['compactness_se'])
        concavity_se = float(request.form['concavity_se'])
        concave_points_se = float(request.form['concave_points_se'])
        symmetry_se = float(request.form['symmetry_se'])
        fractal_dimension_se = float(request.form['fractal_dimension_se'])
        radius_worst = float(request.form['radius_worst'])
        texture_worst = float(request.form['texture_worst'])
        perimeter_worst = float(request.form['perimeter_worst'])
        area_worst = float(request.form['area_worst'])
        smoothness_worst = float(request.form['smoothness_worst'])
        compactness_worst = float(request.form['compactness_worst'])
        concavity_worst = float(request.form['concavity_worst'])
        concave_points_worst = float(request.form['concave_points_worst'])
        symmetry_worst = float(request.form['symmetry_worst'])
        fractal_dimension_worst = float(request.form['fractal_dimension_worst'])

        model = pickle.load(open('cancer_model.pkl', 'rb'))

        prediction = model.predict([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
                                    radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst,
                                    texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]])

        if prediction == 0:
            prediction = 'Benign'
        else:
            prediction = 'Malignant'

        return render_template('cancer.html', prediction_text = 'Cancer is of type: {}.'.format(prediction))
    else:
        return render_template('cancer.html')

@app.route('/diabetes/predict', methods=['GET', 'POST'])
def diabetes_predict():
    if request.method == 'POST':
        pregnancies = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bp'])
        skin_thickness = int(request.form['skin_thickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        model = pickle.load(open('diabetes_model.pkl', 'rb'))

        prediction = model.predict([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])

        if prediction == 0:
            prediction = 'Congrats! You are not Diabetic!'
        else:
            prediction = 'Sorry! You are Diabetic!'

        return render_template('diabetes.html', prediction_text = prediction)
    else:
        return render_template('diabetes.html')

@app.route('/heart/predict', methods=['GET', 'POST'])
def heart_predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        model = pickle.load(open('heart_model.pkl', 'rb'))

        prediction = model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if prediction == 0:
            prediction = "Congrats! You don't have a Heart Disease!"
        else:
            prediction = 'Sorry! You have a Heart Disease!'

        return render_template('heart.html', prediction_text = prediction)
    else:
        return render_template('heart.html')

@app.route('/kidney/predict', methods=['POST', 'GET'])
def kidney_predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        bp = float(request.form['bp'])
        sg = float(request.form['sg'])
        al = float(request.form['al'])
        su = float(request.form['su'])
        rbc = float(request.form['rbc'])
        pc = float(request.form['pc'])
        pcc = float(request.form['pcc'])
        ba = float(request.form['ba'])
        bgr = float(request.form['bgr'])
        bu = float(request.form['bu'])
        sc = float(request.form['sc'])
        sod = float(request.form['sod'])
        pot = float(request.form['pot'])
        hemo = float(request.form['hemo'])
        pcv = float(request.form['pcv'])
        wc = float(request.form['wc'])
        rc = float(request.form['rc'])
        htn = float(request.form['htn'])
        dm = float(request.form['dm'])
        cad = float(request.form['cad'])
        appet = float(request.form['appet'])
        pe = float(request.form['pe'])
        ane = float(request.form['ane'])

        model = pickle.load(open('kidney_model.pkl', 'rb'))

        prediction = model.predict([[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]])

        if prediction == 0:
            prediction = "Congrats! You don't have a Chronic Kidney Disease!"
        else:
            prediction = 'Sorry! You have a Chronic Kidney Disease!'

        return render_template('kidney.html', prediction_text = prediction)
    else:
        return render_template('kidney.html')

@app.route('/liver/predict', methods=['GET', 'POST'])
def liver_predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        tb = float(request.form['tb'])
        db = float(request.form['db'])
        ap = float(request.form['ap'])
        aa = float(request.form['aa'])
        asa = float(request.form['asa'])
        tp = float(request.form['tp'])
        al = float(request.form['al'])
        agr = float(request.form['agr'])

        model = pickle.load(open('liver_model.pkl', 'rb'))

        prediction = model.predict([[age, sex, tb, db, ap, aa, asa, tp, al, agr]])

        if prediction == 0:
            prediction = "Congrats! You don't have a Liver Disease!"
        else:
            prediction = 'Sorry! You have a Liver Disease!'

        return render_template('liver.html', prediction_text = prediction)
    else:
        return render_template('liver.html')

def detect(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x/=255
    x = np.expand_dims(x, axis=0)
    prediction = np.argmax(model.predict(x), axis=1)
    return prediction

@app.route('/malaria/predict', methods=['GET', 'POST'])
def malaria_predict():

    model = load_model('malaria_model')

    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', 'malaria', secure_filename(f.filename))
        f.save(filepath)

        prediction = detect(filepath, model)
        if prediction == 0:
            return "The cell is Parasitized! It's Malaria!"
        else:
            return "The cell is Uninfected! You are Safe!"

@app.route('/pneumonia/predict', methods=['GET', 'POST'])
def pneumonia_predict():

    model = load_model('pneumonia_model')

    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', 'pneumonia', secure_filename(f.filename))
        f.save(filepath)

        prediction = detect(filepath, model)
        if prediction == 0:
            return "CONGRATS! It's all NORMAL!"
        else:
            return "SORRY! It's Pneumonia!"

if __name__ == '__main__':
    app.run(debug=True)
