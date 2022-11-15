import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize, pyramid_reduce
import PIL
from PIL import Image


from flask import Flask, render_template, Response

#Initialize the Flask app
app = Flask(__name__)

# functions associated with model
def prediction(pred):
    return(chr(pred+ 65))


def keras_predict(model, image):
    data = np.asarray( image, dtype="int32" )
    
    pred_probab = model.predict(data)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def keras_process_image(img):
    
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (1,28,28), interpolation = cv2.INTER_AREA)
  
    return img
 

def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

# load the model
model = load_model('CNNmodel.h5')

# define a video capture object
def gen_frames():
	vid = cv2.VideoCapture(0)

	while(True):

		# Capture the video frame
		# by frame
		ret, frame = vid.read() # read the camera frame
		if not ret:
			break
		else:
			frame = cv2.flip(frame,1)

			# Display the resulting frame
			im2 = crop_image(frame, 400,150,200,200)
			image_grayscale = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
			image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (15,15), 0)
			im3 = cv2.resize(image_grayscale_blurred, (28,28), interpolation = cv2.INTER_AREA)
			im4 = np.resize(im3, (28, 28, 1))
			im5 = np.expand_dims(im4, axis=0)
			pred_probab, pred_class = keras_predict(model, im5)
			curr = prediction(pred_class)

			cv2.putText(frame, curr, (100, 300), cv2.FONT_HERSHEY_COMPLEX, 4.0, (0, 255, 255), lineType=cv2.LINE_AA)
			cv2.rectangle(frame, (400, 150), (600, 350), (255, 255, 00), 3)
			
			ret, buffer = cv2.imencode('.jpg', frame)
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
	               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()