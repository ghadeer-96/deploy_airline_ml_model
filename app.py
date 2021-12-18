import flask
import pickle
import pandas as pd
import numpy as np
import sklearn



# Use pickle to load in the pre-trained model
model = pickle.load(open(f'models/desicion_tree_model.sav', 'rb'))


# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')



# Set up the main route
@app.route('/', methods=['GET'])#, 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('home.html'))

@app.route('/predict',methods=['POST'])
def predict():
    cols = ['Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
       'Inflight wifi service', 'Departure/Arrival time convenient',
       'Ease of Online booking', 'Gate location', 'Food and drink',
       'Online boarding', 'Seat comfort', 'Inflight entertainment',
       'On-board service', 'Leg room service', 'Baggage handling',
       'Checkin service', 'Inflight service', 'Cleanliness',
       'Departure Delay in Minutes', 'Arrival Delay in Minutes']   
    input = []
    # Extract the input
    for i in cols:
       input.append(flask.request.form[i])
    
    final = np.array([input])
    final = final.astype('int')
    # Get the model's prediction
    prediction = model.predict(final)[0]
	if prediction:
		return flask.render_template('home.html',pred='satisfied :)')
	else:
		return flask.render_template('home.html',pred='not satisfied :(')

    

##################################
if __name__ == '__main__':
    app.run()

