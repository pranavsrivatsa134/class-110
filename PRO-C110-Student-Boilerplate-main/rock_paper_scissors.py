# import the opencv library
import cv2
import numpy as np
import tensorFlow as tf
model = tf.keras.model.loan_model('keras_madel.h5')

# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    image = cv2.resize(frame,(224,224))
    test_image = np.array(img,dtype = np.float32)
    test_image = np.expand_dims(test_image,axis = 0)
    normalizedImage = test_image/255.0
    prediction = model.predict(normalizedImage)
      
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()