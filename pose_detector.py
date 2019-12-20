import numpy as np
import time
import os

from flask import jsonify
from cv2 import cv2


protoFile = "pose-detection/pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose-detection/pose/coco/pose_iter_440000.caffemodel"

BODY_DICT = {'nose': 0,
    'neck': 1, 'shoulder_right': 2, 'elbow_right': 3, 'wrist_right': 4, 'shoulder_left': 5,
    'elbow_left': 6, 'wrist_left': 7, 'hip_right': 8, 'knee_right': 9, 'ankle_right':10,
    'hip_left': 11, 'knee_left': 12, 'ankle_left': 13, 'eyebrow_right': 14, 'eyebrow_left': 15,
    'ear_right': 16, 'ear_left': 17
}

BODY_DICTIONNARY_2 = {'0': 'nose',
    '1': 'neck', '2': 'shoulder_right', '3': 'elbow_right', '4': 'wrist_right', '5': 'shoulder_left',
    '6': 'elbow_left', '7': 'wrist_left', '8': 'hip_right', '9': 'knee_right', '10': 'ankle_right',
    '11': 'hip_left', '12': 'knee_left', '13': 'ankle_left', '14': 'eyebrow_right', '15': 'eyebrow_left',
    '16': 'ear_right', '17': 'ear_left'
}

nPoints = 18
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17] ]
POSE_HEAD = [ [BODY_DICT['ear_right'], BODY_DICT['ear_left']], [BODY_DICT['ear_left'], BODY_DICT['neck']], [BODY_DICT['neck'], BODY_DICT['ear_right']] ]

x1_head = [2, 8, 9, 10, 15]
y1_head = [0, 13, 14, 15, 16]
x2_head = [5, 11, 12, 13, 16]
y2_head = [1, 2, 3, 4, 5, 6, 17]

x1_upper = [2, 3, 4, 8, 15]
y1_upper = [0, 1, 13, 14, 15 ,16]
x2_upper = [5, 6, 11, 16, 17]
y2_upper = [8, 9, 11, 12]

x1_lower = [8, 9, 10]
y1_lower = [8, 11]
x2_lower = [11, 12, 13]
y2_lower = [10, 13]

points_list = [x1_head, x1_upper, x1_lower, x2_head, x2_upper, x2_lower, y1_head, y1_upper, y1_lower, y2_head, y2_upper, y2_lower]

def pose_main(myImagePath, imageName):
    """detect pose"""

    #data = request.json
    #myPose = data["myPose"]

    print("will read image")
    frame = cv2.imread(myImagePath)
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    t = time.time()
    # input image dimensions for the network
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                            (0, 0, 0), swapRB=False, crop=False)

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    net.setInput(inpBlob)

    output = net.forward()
    print("shape: ", output.shape)
    print("time taken by network : {:.3f}".format(time.time() - t))

    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        #print("Point " + str(i) + ": ")
        #print("Prob: ", prob)
        #print("x: " + str(x))
        #print("y: " + str(y))
        if prob > threshold : 
            #cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            #cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)
    print("points: ", points)

    # Draw Skeleton
    #for pair in POSE_PAIRS:
        #partA = pair[0]
        #partB = pair[1]

        #if points[partA] and points[partB]:
            #cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            #cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    print("points_list: ", points_list)
    for num, corner in enumerate(points_list):
        print(corner)
        for i, j in enumerate(corner):
            if (points[j] is None):
                corner[i] = "NA"
            else:
                corner[i] = points[j]
        print("corner: ", corner)
        #corner = [loc for loc in corner if loc != "NA"]
        points_list[num] = [loc for loc in corner if loc != "NA"]
        print(num)
        print("corner: ", points_list[num])

    print("points_list: ", points_list)
    print("len: ", len(points_list))

    print("x1_list")
    for num, corner in enumerate(points_list[0:3]):
        print(corner)
        if corner == []:
            corner = 0
        else:
            for i, biPoint in enumerate(corner):
                corner[i] = biPoint[0]
            #corner = #int(round(min(corner) * 0.75, 0)) # expansion to get a wider frame
            corner = min(corner)
        points_list[num] = corner
    
    print("points_list: ", points_list)
    print("len: ", len(points_list))    

    print("x2_list")
    print(points_list[3:5])
    for num, corner in enumerate(points_list[3:6]):
        print("points_list[3:5]: ", points_list[3:6])
        print(num)
        print(corner)
        if corner == []:
            corner = frameWidth
        else:
            for i, biPoint in enumerate(corner):
                corner[i] = biPoint[0]
            #corner = min(max(corner) * 1.1, frameWidth) # expansion to get a wider frame
            corner = max(corner)
        points_list[num + 3] = corner

    print("y1_list")
    for num, corner in enumerate(points_list[6:9]):
        if corner == []:
            corner = 0
        else:
            for i, biPoint in enumerate(corner):
                corner[i] = biPoint[0]
            corner = min(corner) #* 0.9 # expansion to get a wider frame
        points_list[num + 6] = corner

    print("y2_list")
    for num, corner in enumerate(points_list[9:]):
        if corner == []:
            corner = frameHeight
        else:
            for i, biPoint in enumerate(corner):
                corner[i] = biPoint[0]
            #corner = max(max(corner) * 1.1, frameHeight) # expansion to get a wider frame
            corner = max(corner)
        points_list[num + 9] = corner

    print("points_list: ", points_list)
    print("len: ", len(points_list))    

    points_dict = {
        "x1_head": points_list[0], "x1_upper": points_list[1], "x1_lower": points_list[2],\
        "x2_head": points_list[3], "x2_upper": points_list[4], "x2_lower": points_list[5],\
        "y1_head": points_list[6], "y1_upper": points_list[7], "y1_lower": points_list[8],\
        "y2_head": points_list[9], "y2_upper": points_list[10], "y2_lower": points_list[11]
        }

    print(points_dict)
    print("will draw")

    cv2.line(frame, (points_dict['x1_head'], points_dict['y1_head']), (points_dict['x2_head'], points_dict['y1_head']), (0, 255, 255), 2)
    cv2.line(frame, (points_dict['x2_head'], points_dict['y1_head']), (points_dict['x2_head'], points_dict['y2_head']), (0, 255, 255), 2)
    cv2.line(frame, (points_dict['x2_head'], points_dict['y2_head']), (points_dict['x1_head'], points_dict['y2_head']), (0, 255, 255), 2)
    cv2.line(frame, (points_dict['x1_head'], points_dict['y2_head']), (points_dict['x1_head'], points_dict['y1_head']), (0, 255, 255), 2)
    print("passed head")
    cv2.line(frame, (points_dict['x1_upper'], points_dict['y1_upper']), (points_dict['x2_upper'], points_dict['y1_upper']), (255, 0, 255), 2)
    cv2.line(frame, (points_dict['x2_upper'], points_dict['y1_upper']), (points_dict['x2_upper'], points_dict['y2_upper']), (255, 0, 255), 2)
    cv2.line(frame, (points_dict['x2_upper'], points_dict['y2_upper']), (points_dict['x1_upper'], points_dict['y2_upper']), (255, 0, 255), 2)
    cv2.line(frame, (points_dict['x1_upper'], points_dict['y2_upper']), (points_dict['x1_upper'], points_dict['y1_upper']), (255, 0, 255), 2)
    print("passed upper")
    cv2.line(frame, (points_dict['x1_lower'], points_dict['y1_lower']), (points_dict['x2_lower'], points_dict['y1_lower']), (255, 255, 0), 2)
    cv2.line(frame, (points_dict['x2_lower'], points_dict['y1_lower']), (points_dict['x2_lower'], points_dict['y2_lower']), (255, 255, 0), 2)
    cv2.line(frame, (points_dict['x2_lower'], points_dict['y2_lower']), (points_dict['x1_lower'], points_dict['y2_lower']), (255, 255, 0), 2)
    cv2.line(frame, (points_dict['x1_lower'], points_dict['y2_lower']), (points_dict['x1_lower'], points_dict['y1_lower']), (255, 255, 0), 2)
    print("passed lower")

    #crop_img_head = frame[x1_head: x2_head, y1_head: y2_head]
    #cv2.imshow("crop_img_head", crop_img_head)

    print("Image saved")
    #cv2.imshow('Output-Keypoints', frameCopy)
    #cv2.imshow('Output-Skeleton', frame)
    print("first show")
    head_image = frameCopy[points_dict['y1_head']: points_dict['y2_head'], points_dict['x1_head']: points_dict['x2_head']]
    upper_image = frameCopy[points_dict['y1_upper']: points_dict['y2_upper'], points_dict['x1_upper']: points_dict['x2_upper']]
    lower_image = frameCopy[points_dict['y1_lower']: points_dict['y2_lower'], points_dict['x1_lower']: points_dict['x2_lower']]

    #cv2.imshow('Output-Head', head_image)
    #cv2.imshow('Output-Upper', upper_image)
    #cv2.imshow('Output-Lower', lower_image)
    cv2.imwrite(os.path.join("./static/uploaded_pictures" , str(imageName) + '_Whole.jpg'), frameCopy)
    cv2.imwrite(os.path.join("./static/uploaded_pictures" , str(imageName) + '_Head.jpg'), head_image)
    cv2.imwrite(os.path.join("./static/uploaded_pictures" , str(imageName) +' _Upper.jpg'), upper_image)
    cv2.imwrite(os.path.join("./static/uploaded_pictures" , str(imageName) +' _Lower.jpg'), lower_image)

    #cv2.imwrite('./client_images/images_output/Output-Keypoints.jpg', frameCopy)
    #cv2.imwrite('./client_images/images_output/Output-Skeleton.jpg', frame)

    print("Total time taken : {:.3f}".format(time.time() - t))

    #cv2.waitKey(0)

    #return jsonify({"status": "success"})
    return "success"