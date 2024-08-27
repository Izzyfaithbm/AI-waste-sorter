# This is my innovation for my science summative!!
# I'll provide more info in my report, but as a short summary: this program
# detects objects from your webcam and identifies them as either trash,
# recyclables or compost. Red text provides further explanations (enclosed in """) 

""" downloads opencv, image processing/computer vision open-source algorithms (for object detection) i also installed numpy, ultralytics and pillow """

!pip install opencv-python

""" import libraries and APIs (large arrays, object detection, video camera) """
import numpy as np  
from ultralytics import YOLO
import cv2
from PIL import Image

""" builds the 3 categories from objects the model can detect """ 
compost = ['cake', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut']
trash = ['toothbrush', 'cup', 'fork', 'knife', 'spoon', 'bowl']
recycling = ['book', 'vase', 'bottle', 'wine glass']
 
""" extract YOLO model (medium & detection kind) """
model = YOLO("yolov8m.pt")

""" extracts my laptop's camera """ 
camera = cv2.VideoCapture(0)  

""" loop starts immediately and repeats until forever/"break" since there's no
condition besides True; this loop will repeat for each frame it captures """
while True:
    
    """ while loop will keep storing each camera frame 
    ret (if ret == True, basically) True meaning the code was  
    able to read a frame from the camera. Otherwise, ret would be False 
    and the loop would break """
    ret, frame = camera.read()

    """ model predicts bounding boxes, confidence scores, and class labels 
    and stores in results variable:
    Bounding boxes - box surrounding the object
    Confidence scores - how confident the robot is 
    (0.95 means 95% sure that's a dog, 0.20 means it's 20% sure... which is bad!)
    Class labels - the class the object sorts in (ex. dog, person, toothbrush) """
    results = model.predict(frame)

    """ runs if results variable is True (AKA the model can predict a webcam frame captured) """
    if results and len(results) > 0:
        
        """ multiple results could be detected, so it extracts the first (boxes) """
        result = results[0]

        """ verifies there is an object being detected """ 
        if len(result.boxes) > 0:
            
            """ multiple boxes (objects) could be detected, so it extracts the first one """
            box = result.boxes[0]

            """ stores confidence level """ 
            for box in result.boxes:
                confidence = box.conf[0].item()

            """ only detects objects with confidence levels over 70% """
            if confidence > 0.4:
                
                """ stores object type """
                object = result.names[box.cls[0].item()]
                    
                """ sorts object to its category from list and plots label on it 
                colour written in BGR format for cv2"""
                if object in compost:
                    label = "COMPOST!"
                    colour = (144, 238, 144)
                elif object in trash:
                    label = "TRASH!"
                    colour = (226, 228, 229)
                elif object in recycling:
                    label = "RECYCLING!"
                    colour = (255, 150, 0)
                else:
                    label = "NON-WASTE"
                    colour = (0, 192, 255)
        
                """ draws textbox and text on frame """
                cv2.rectangle(frame, (100, 10), (550, 50), colour, -1) 
                cv2.putText(frame, ("DETECTED: " + label), (130, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA) 
                
                """ converts BGR (opencv/detection) to RGB (pil/image)
                and stores inside img variable so it can take image from 
                model's array (results) and plot detections on screen's frame. """ 
                img = Image.fromarray(result.plot()[:, :, ::-1])
                """ converts back to BGR and array for cv2 (opencv) use """
                img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)   
                """ displays frame with detections, names window ECOLIFE """
                cv2.imshow("ECOLIFE", img_cv2)

    """ waits 10 milliseconds and checks if q is pressed through its ASCII value """
    if cv2.waitKey(10) & 0xFF == ord('q'):

        """ ends loop if q is pressed """
        break  

""" occurs when loop ends
manages camera resources """
camera.release()

""" closes window """
cv2.destroyAllWindows()
