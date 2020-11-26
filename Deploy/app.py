#Important Modules
from flask import Flask,render_template, url_for ,flash , redirect
#from forms import RegistrationForm, LoginForm
import joblib
from flask import request
import numpy as np
import tensorflow
#from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
#from flask_sqlalchemy import SQLAlchemy
#from model_class import DiabetesCheck, CancerCheck

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
#from tensorflow.keras.layers import GlobalMaxPooling2D, Activation
#from tensorflow.keras.layers.normalization import BatchNormalization
#from tensorflow.keras.layers.merge import Concatenate
#from tensorflow.keras.models import Model

import os
from flask import send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
#from this import SQLAlchemy
app=Flask(__name__,template_folder='template')



# RELATED TO THE SQL DATABASE
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
#app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
#db=SQLAlchemy(app)

#from model import User,Post

#//////////////////////////////////////////////////////////

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

#graph = tf.get_default_graph()
#with graph.as_default():;
from tensorflow.keras.models import load_model

model = load_model('model.h5')
#model = load_model('new_model.h5')

#FOR THE FIRST MODEL

# call model to predict an image
def api(full_path):
    data = image.load_img(full_path, target_size=(64, 64, 3),)
    
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model.predict(data)
    return predicted


# home page

#@app.route('/')
#def home():
 #  return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'Infected', 1: 'Uninfected'}
            result = api(full_name)
            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")
            return redirect(url_for("Malaria"))

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)





@app.route("/home")
def home():
	return render_template("home.html")

@app.route("/")
def Malaria():
    return render_template("index.html")




if __name__ == "__main__":
	app.run(debug=True, port=5001)
