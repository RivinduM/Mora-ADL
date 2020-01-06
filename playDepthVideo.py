import argparse

import cv2
import numpy as np


def parse_arg():
    parser = argparse.ArgumentParser(description='Test OpenNI2 and NiTE2.')
    parser.add_argument('-w', '--window_width', type=int, default=1024,
                        help='Specify the window width.')
    return parser.parse_args()


def getSequence(fileName):
    Frames = np.fromfile(fileName, dtype="uint16")
    numOfFrames = int(Frames.shape[0]/(640*480))
    return Frames.reshape((numOfFrames, 480, 640))


def readSkeltonFile(filePath):
    skeltons = []
    file = open(filePath, 'r')
    lines = file.readlines()
    lineNumber = 0
    while lineNumber < len(lines):
        line = lines[lineNumber]
        if line == "===\n":
            skeltons.append([])
            lineNumber += 1
        else:
            joints = []
            for i in range(lineNumber, lineNumber+15):
                joints.append([float(x) for x in lines[i].split(', ')])
            lineNumber += 15
            skeltons.append(joints)
    return skeltons


def drawSkelton(skeleton, img):
    col = (255, 0, 0)
    for joint in skeleton:
        cv2.circle(img, (int(joint[3]), int(joint[4])), 2, col, -1)

    cv2.line(img, (int(skeleton[0][3]), int(skeleton[0][4])), (int(
        skeleton[1][3]), int(skeleton[1][4])), col, 1)
    cv2.line(img, (int(skeleton[1][3]), int(skeleton[1][4])), (int(
        skeleton[2][3]), int(skeleton[2][4])), col, 1)
    cv2.line(img, (int(skeleton[1][3]), int(skeleton[1][4])), (int(
        skeleton[3][3]), int(skeleton[3][4])), col, 1)
    cv2.line(img, (int(skeleton[2][3]), int(skeleton[2][4])), (int(
        skeleton[8][3]), int(skeleton[8][4])), col, 1)
    cv2.line(img, (int(skeleton[3][3]), int(skeleton[3][4])), (int(
        skeleton[8][3]), int(skeleton[8][4])), col, 1)
    cv2.line(img, (int(skeleton[2][3]), int(skeleton[2][4])), (int(
        skeleton[4][3]), int(skeleton[4][4])), col, 1)
    cv2.line(img, (int(skeleton[6][3]), int(skeleton[6][4])), (int(
        skeleton[4][3]), int(skeleton[4][4])), col, 1)
    cv2.line(img, (int(skeleton[5][3]), int(skeleton[5][4])), (int(
        skeleton[3][3]), int(skeleton[3][4])), col, 1)
    cv2.line(img, (int(skeleton[5][3]), int(skeleton[5][4])), (int(
        skeleton[7][3]), int(skeleton[7][4])), col, 1)
    cv2.line(img, (int(skeleton[8][3]), int(skeleton[8][4])), (int(
        skeleton[9][3]), int(skeleton[9][4])), col, 1)
    cv2.line(img, (int(skeleton[8][3]), int(skeleton[8][4])), (int(
        skeleton[10][3]), int(skeleton[10][4])), col, 1)
    cv2.line(img, (int(skeleton[9][3]), int(skeleton[9][4])), (int(
        skeleton[11][3]), int(skeleton[11][4])), col, 1)
    cv2.line(img, (int(skeleton[13][3]), int(skeleton[13][4])), (int(
        skeleton[11][3]), int(skeleton[11][4])), col, 1)
    cv2.line(img, (int(skeleton[12][3]), int(skeleton[12][4])), (int(
        skeleton[10][3]), int(skeleton[10][4])), col, 1)
    cv2.line(img, (int(skeleton[12][3]), int(skeleton[12][4])), (int(
        skeleton[14][3]), int(skeleton[14][4])), col, 1)


def playVideo(frames, skeltons, silhouette=False):
    for frameNum in range(int(frames.shape[0]/2)):
        userFrame = frames[2*frameNum+1]
        maxId = np.amax(userFrame)
        img = frames[2*frameNum]
        if maxId > 0 and silhouette:
            userFrame = userFrame/maxId
            img = np.multiply(img, userFrame)
        img = img.astype(np.float32)
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(img)
        if (min_val < max_val):
            img = (img - min_val) / (max_val - min_val)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if skeltons[frameNum] != []:
            drawSkelton(skeltons[frameNum], img)
        cv2.imshow("Depth", cv2.resize(img, (1024, int(480*1024/640))))
        if (cv2.waitKey(7) & 0xFF == ord('q')):
            break
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-d', '--depthPath', type=str,
    help='depth file path')
parser.add_argument(
    '-s', '--skelPath', type=str,
    help='skelelton file path')
parser.add_argument(
    '-sil', '--silhouette', type=str,
    help='silhouette True/False')

args = parser.parse_args()

if (args.depthPath == None):
    parser.exit(
        "Depth file path not specified!! Pass depth file path with -d argument")
elif (args.skelPath == None):
    parser.exit(
        "Skeleton file path not specified!! Pass skeleton file path with -s argument")
else:
    Frames = getSequence(args.depthPath)
    skeltons = readSkeltonFile(args.skelPath)

    playVideo(Frames, skeltons, args.silhouette)
