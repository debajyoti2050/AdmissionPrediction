import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('lr_model.pickle', 'rb'))


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['GET','post'])
def predict():
	
	GRE_Score = int(request.form['GRE Score'])
	TOEFL_Score = int(request.form['TOEFL Score'])
	
	CGPA = float(request.form['CGPA'])
	
	
	final_features = pd.DataFrame([[GRE_Score, TOEFL_Score, CGPA]])
	
	predict = model.predict(final_features)
	
	output = {predict[]*100:.2f}
	
	return render_template('index.html', prediction_text='Admission chances are {}'.format(output))
	
if __name__ == "__main__":
	app.run(debug=True)
