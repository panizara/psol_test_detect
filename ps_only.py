import cv2
import numpy as np
import winsound


def ob():
    net = cv2.dnn.readNet('yolov3.weights' , 'yolov3.cfg')

    classes = []
    #coco.names
    #pam.names
    with open("coco.names", "r") as f:
        classes = f.read().splitlines()

    #cap = cv2.VideoCapture('rtsp://158.108.118.90')172.20.0.1
    # cap = cv2.VideoCapture('videoplayback.mp4')
    #cap = cv2.VideoCapture('rtsp://10.37.52.126')
    #cap = cv2.VideoCapture('rtsp://192.168.1.7')
    cap = cv2.VideoCapture(0)
    #cap = cv2.imread('humanchildren1.jpg')bottle
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(50, 3))

    while True:
        def rescale_frame(frame, percent=75):
            scale_percent = 75
            width = int(frame.shape[1] * scale_percent/ 100)
            height = int(frame.shape[0] * scale_percent/ 100)
            dim = (width, height)
            return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
        
        #_, frame = cap.read()
        ret, frame = cap.read()
        frame = rescale_frame(frame, percent=75)

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)
        boxes = []
        confidences = []
        class_ids = []
        obj_count = []


        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
        #for i in range(len(class_ids[0])):
        #    if class_IDs[0][i].asscalar() != 17.:
        #        scores[0][i] = 0

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                #res = str(round(label[i]),2)
                color = colors[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                #cv2.putText(img, label + "xxx " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
                #cv2.putText(img, "something: " + str(round(obj_count, 2)), (10, 50), font, 2, (0, 0,255), 3)
                cv2.putText(frame, label + " " +confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
                
                if label == "person" :
                    
                    print ('Meet person in the car')
                    cv2.imshow('person',frame)
                    winsound.Beep(500,2000)
        
        
        cv2.imshow('frame', frame)
        #cv2.putText(frame, "Number of Objects: " + str(round(obj_count, 2)), (10, 50), font, 2, (0, 0,255), 3)
        #winsound.Beep(500, 3000)
        if cv2.waitKey(20)& 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


ob()
