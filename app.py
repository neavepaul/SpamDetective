from flask import Flask,render_template,url_for,request
import joblib


app = Flask(__name__)

model = joblib.load('spam_classifier_model.pkl')
cv = joblib.load('count_vectorizer.pkl')

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	message = request.form['message']
	data = [message]
	vect = cv.transform(data).toarray()
	my_prediction = model.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)