import cv2 
import numpy as np 
 
confThreshold = 0.6 
 
cap = cv2.VideoCapture(0) 
 
# Load class names 
classesFile = 'coco80.names' 
classes = [] 
with open(classesFile, 'r') as f: 
    classes = f.read().splitlines() 
 
# Load YOLO model 
net = cv2.dnn.readNetFromDarknet('yolov3-608.cfg', 'yolov3-608.weights') 
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) 
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) 
 
# Define fruit prices 
fruit_prices = {"apple": 1.0, "banana": 0.5, "orange": 0.75} 
 
while True: 
    success, img = cap.read() 
    height, width, ch = img.shape 
 
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False) 
    net.setInput(blob) 
 
    output_layers_names = net.getUnconnectedOutLayersNames() 
    LayerOutputs = net.forward(output_layers_names) 
 
    bboxes = [] 
    confidences = [] 
    class_ids = [] 
 
    for output in LayerOutputs: 
        for detection in output: 
            scores = detection[5:] 
            class_id = np.argmax(scores) 
            confidence = scores[class_id] 
            if confidence > 0.8 and classes[class_id] in fruit_prices: 
                center_x = int(detection[0] * width) 
                center_y = int(detection[1] * height) 
                w = int(detection[2] * width) 
                h = int(detection[3] * height) 
                x = int(center_x - w / 2) 
                y = int(center_y - h / 2) 
 
                bboxes.append([x, y, w, h]) 
                confidences.append(float(confidence)) 
                class_ids.append(class_id) 
 
    indexes = cv2.dnn.NMSBoxes(bboxes, confidences, confThreshold, 0.4) 
 
    # Reset total price and detected fruits 
    detected_fruits = {} 
    total_price = 0 
 
    if len(indexes) > 0: 
        for i in indexes.flatten(): 
            x, y, w, h = bboxes[i] 
            label = str(classes[class_ids[i]]) 
            confidence = confidences[i] 
            price = fruit_prices.get(label, 0) 
 
            # Update detected fruits count 
            if label in detected_fruits: 
                detected_fruits[label] += 1 
            else: 
                detected_fruits[label] = 1 
 
            # Update total price only once per unique detection 
            total_price += price 
 
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 
 
    # Calculate total fruits and display 
    total_fruits = sum(detected_fruits.values()) 
    cv2.putText(img, f"Total Fruits: {total_fruits}", (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
    cv2.putText(img, f"Total Price: ${total_price:.2f}", (width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
 
    cv2.imshow('Image', img) 
    if cv2.waitKey(1) & 0xff == ord('q'): 
        break 
 
cap.release() 
cv2.destroyAllWindows() 
