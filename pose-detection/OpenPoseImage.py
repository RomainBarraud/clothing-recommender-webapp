from flask import Flask, request, jsonify
import time
import numpy as np

from cv2 import cv2

protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
nPoints = 18
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]


app = Flask(__name__)


@app.route("/", methods=["GET","POST"])
def pose_main():
    """detect pose"""

    #data = request.json
    #myPose = data["myPose"]

    print("will read image")
    frame = cv2.imread("single.jpeg")
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

    net.setInput(inpBlob)

    output = net.forward()
    print(output.shape)
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

        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


    #cv2.imshow('Output-Keypoints', frameCopy)
    #cv2.imshow('Output-Skeleton', frame)

    cv2.imwrite('Output-Keypoints.jpg', frameCopy)
    cv2.imwrite('Output-Skeleton.jpg', frame)

    print("Total time taken : {:.3f}".format(time.time() - t))

    #cv2.waitKey(0)

    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.debug = True
    app.run(host="127.0.0.1", port=5001)


