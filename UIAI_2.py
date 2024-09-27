from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import urllib.request
import os
from werkzeug.utils import secure_filename
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
 
app = Flask(__name__)

train_dir = "../ForTrain"
valid_dir = "../ForValid"

# Updated Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=20,         # New - rotation augmentation
    horizontal_flip=True,      # New - horizontal flipping
    fill_mode='nearest'
)
batch_size = 32
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

model = load_model("TrainedTest5.keras", custom_objects={})
class_dict = train_generator.class_indices
li = list(class_dict.keys())

UPLOAD_FOLDER = 'static/uploads'
app.secret_key = "secret key"

# Get the absolute path to the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Set upload folder and file size limit
UPLOAD_FOLDER = os.path.join(base_dir, 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Max size file 5 MB

# Allowed extensions for image files
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'heif'])

# Validate file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')  # Render homepage

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No image selected for prediction')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Save image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        # Load and process image
        new_img = image.load_img(image_path, target_size=(224, 224))
        img = image.img_to_array(new_img)
        img = np.expand_dims(img, axis=0)
        img = img / 255

        # Predict using the model
        prediction = model.predict(img)
        d = prediction.flatten()
        j = d.max()
        
        for index, item in enumerate(d):
            if item == j:
                class_name = li[index]
                new_class_name = class_name.replace("_", " ").title()

        # # Plot image with predicted class name
        # plt.figure(figsize=(4, 4))
        # plt.imshow(new_img)
        # plt.axis('off')
        # plt.title(new_class_name)
        # plt.show()

        flash('Prediction: ' + new_class_name)  # Show prediction result

        # Return the page with the uploaded image and prediction
        return render_template('index.html', filename=filename, prediction=new_class_name)
    else:
        flash('Allowed image types are - png, jpg, jpeg, heif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# To display the uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return render_template('display.html', filename=filename)

if __name__ == "__main__":
    app.run(debug=True)
