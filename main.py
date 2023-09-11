import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *

model=YOLO('yolov8s.pt') # pre-trained model --> s = small

cap = cv2.VideoCapture('veh2.mp4')

# access file object class list
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)

tracker = Tracker() # define tracker from modul tracker.py

while True:
  _,frame = cap.read() # define window video frame

  frame = cv2.resize(frame,(1020,500)) # size frame
  results = model.predict(frame) # prediction results of frame
  # print(results)

  a = results[0].boxes.data # list of predictions
  # print(a)
  px = pd.DataFrame(a).astype("float") # arrange prediction object to table
  # print(px)

  list=[] # container for detected car in object id

  for index,row in px.iterrows():
    # print(row)
    y1 = int(row[1]) # top
    x1 = int(row[0]) # left
    x2 = int(row[2]) # right
    y2 = int(row[3]) # bottom

    d = int(row[5])
    c = class_list[d] # object class

    if 'car' in c: # filter object -- only car
      list.append([x1,y1,x2,y2]) # add car to container

    bbox_id = tracker.update(list) # assign id and position object (car)
    for bbox in bbox_id:
      x3, y3, x4, y4, id = bbox # get coordinate and id
      cx = int(x3+x4)//2 # get x coordinate center
      cy = int(y3+y4)//2 # get y coordinate center
      cv2.circle(frame,(cx,cy),4,(0,0,255),-1) # object dot
      cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # show object id

  # show window frame
  cv2.imshow("RGB", frame)
  if cv2.waitKey(1)&0xFF==27: # press Esc for quit
    break

cap.release()
cv2.destroyAllWindows()