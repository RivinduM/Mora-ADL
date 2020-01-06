import argparse
import ctypes
import datetime
import os
import queue as queue
import sys
import threading
import time

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf

from openni import nite2, openni2, utils


class KinectRecoder(object):

    def __init__(self, dirPath):
        self.record = False
        self.active = True
        self.GRAY_COLOR = (64, 64, 64)
        self.col = (255, 0, 0)
        self.CAPTURE_SIZE_KINECT = (512, 424)
        self.CAPTURE_SIZE_OTHERS = (640, 480)
        self.dirPath = dirPath
        self.depthImages = []
        self.skeltons = []
        self.videos = []
        self.recordStartTime = ''
        self.recordFinishTime = ''
        self.frameCount = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (0, 25)
        self.fontScale = 1
        self.fontColor = (0, 0, 255)
        self.lineType = 2
        self.thread = threading.Thread(target=self.run, args=())
        self.thread.start()

    def startTheThread(self):
        self.record = True

    def stopTheThread(self):
        self.record = False
        self.active = False

    def draw_limb(self, img, ut, j1, j2):
        (x1, y1) = ut.convert_joint_coordinates_to_depth(
            j1.position.x, j1.position.y, j1.position.z)
        (x2, y2) = ut.convert_joint_coordinates_to_depth(
            j2.position.x, j2.position.y, j2.position.z)

        if (0.4 < j1.positionConfidence and 0.4 < j2.positionConfidence):
            c = self.GRAY_COLOR if (
                j1.positionConfidence < 1.0 or j2.positionConfidence < 1.0) else self.col
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 1)

            c = self.GRAY_COLOR if (j1.positionConfidence < 1.0) else self.col
            cv2.circle(img, (int(x1), int(y1)), 2, c, -1)

            c = self.GRAY_COLOR if (j2.positionConfidence < 1.0) else self.col
            cv2.circle(img, (int(x2), int(y2)), 2, c, -1)

    def draw_skeleton(self, img, ut, user):
        for idx1, idx2 in [(nite2.JointType.NITE_JOINT_HEAD, nite2.JointType.NITE_JOINT_NECK),
                           # upper body
                           (nite2.JointType.NITE_JOINT_NECK,
                            nite2.JointType.NITE_JOINT_LEFT_SHOULDER),
                           (nite2.JointType.NITE_JOINT_LEFT_SHOULDER,
                            nite2.JointType.NITE_JOINT_TORSO),
                           (nite2.JointType.NITE_JOINT_TORSO,
                            nite2.JointType.NITE_JOINT_RIGHT_SHOULDER),
                           (nite2.JointType.NITE_JOINT_RIGHT_SHOULDER,
                            nite2.JointType.NITE_JOINT_NECK),
                           # left hand
                           (nite2.JointType.NITE_JOINT_LEFT_HAND,
                            nite2.JointType.NITE_JOINT_LEFT_ELBOW),
                           (nite2.JointType.NITE_JOINT_LEFT_ELBOW,
                            nite2.JointType.NITE_JOINT_LEFT_SHOULDER),
                           # right hand
                           (nite2.JointType.NITE_JOINT_RIGHT_HAND,
                            nite2.JointType.NITE_JOINT_RIGHT_ELBOW),
                           (nite2.JointType.NITE_JOINT_RIGHT_ELBOW,
                            nite2.JointType.NITE_JOINT_RIGHT_SHOULDER),
                           # lower body
                           (nite2.JointType.NITE_JOINT_TORSO,
                            nite2.JointType.NITE_JOINT_LEFT_HIP),
                           (nite2.JointType.NITE_JOINT_LEFT_HIP,
                            nite2.JointType.NITE_JOINT_RIGHT_HIP),
                           (nite2.JointType.NITE_JOINT_RIGHT_HIP,
                            nite2.JointType.NITE_JOINT_TORSO),
                           # left leg
                           (nite2.JointType.NITE_JOINT_LEFT_FOOT,
                            nite2.JointType.NITE_JOINT_LEFT_KNEE),
                           (nite2.JointType.NITE_JOINT_LEFT_KNEE,
                            nite2.JointType.NITE_JOINT_LEFT_HIP),
                           # right leg
                           (nite2.JointType.NITE_JOINT_RIGHT_FOOT,
                            nite2.JointType.NITE_JOINT_RIGHT_KNEE),
                           (nite2.JointType.NITE_JOINT_RIGHT_KNEE, nite2.JointType.NITE_JOINT_RIGHT_HIP)]:
            self.draw_limb(
                img, ut, user.skeleton.joints[idx1], user.skeleton.joints[idx2])

    def init_capture_device(self):
        openni2.initialize()
        nite2.initialize()
        return openni2.Device.open_any()

    def close_capture_device(self):
        nite2.unload()
        openni2.unload()

    def run(self):
        dev = self.init_capture_device()
        colorStream = dev.create_color_stream()

        dev_name = dev.get_device_info().name.decode('UTF-8')
        print("Device Name: {}".format(dev_name))
        use_kinect = False
        if dev_name == 'Kinect':
            use_kinect = True
            print('using Kinect.')
        try:
            user_tracker = nite2.UserTracker(dev)
        except utils.NiteError:
            print("Unable to start the NiTE human tracker. Check "
                  "the error messages in the console. Model data "
                  "(s.dat, h.dat...) might be inaccessible.")
            sys.exit(-1)

        (img_w, img_h) = self.CAPTURE_SIZE_KINECT if use_kinect else self.CAPTURE_SIZE_OTHERS
        win_w = 1024
        win_h = int(img_h * win_w / img_w)

        colorStream.start()

        while self.active:
            colorFrame = colorStream.read_frame()
            colorFrameData = colorFrame.get_buffer_as_triplet()
            cimg = np.ndarray((colorFrame.height, colorFrame.width, 3), dtype=np.uint8,
                              buffer=colorFrameData)

            ut_frame = user_tracker.read_frame()

            depth_frame = ut_frame.get_depth_frame()
            depth_frame_data = depth_frame.get_buffer_as_uint16()

            simg = np.ndarray((depth_frame.height, depth_frame.width), dtype=np.uint16,
                              buffer=depth_frame_data)

            img = simg.astype(np.float32)

            (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(img)

            if (min_val < max_val):
                img = (img - min_val) / (max_val - min_val)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            if ut_frame.users:
                for user in ut_frame.users:
                    if user.is_new():
                        print("new human id:{} detected.".format(user.id))
                        user_tracker.start_skeleton_tracking(user.id)
                    elif (user.state == nite2.UserState.NITE_USER_STATE_VISIBLE and
                          user.skeleton.state == nite2.SkeletonState.NITE_SKELETON_TRACKED):
                        self.draw_skeleton(img, user_tracker, user)
                        if self.record:
                            skeltonAdded = True
                            self.addSkelton(user_tracker, user)

        
            if self.record:
                if self.frameCount == 0:
                    self.recordStartTime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
                    self.dirPath = self.dirPath + self.recordStartTime
                    writer = cv2.VideoWriter(
                        self.dirPath + ".avi", cv2.VideoWriter_fourcc(*'XVID'), 25, (640, 480), True)
                self.frameCount += 1
                dirPath = self.dirPath + "__" + \
                    str(self.frameCount)+"__" + \
                    datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
                np.array(simg).tofile(dirPath+"depth.bin")
                writer.write(cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB))
                cv2.putText(img, 'Recording', self.bottomLeftCornerOfText, self.font, self.fontScale,
                            self.fontColor, self.lineType)

            cv2.imshow("Depth", cv2.resize(img, (win_w, win_h)))
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break

        self.recordFinishTime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        self.close_capture_device()
        cv2.destroyAllWindows()
        if self.frameCount > 0:
            print("Writing Started")
            self.dirPath = self.dirPath+'__'+self.recordFinishTime
            print("Writing Finished")
        print("Frame Count : ", self.frameCount)

    def writeIntToSkeltonFile(self, filePath):
        file = open(filePath, 'a')
        for skeltonCordinates in self.skeltons:
            if len(skeltonCordinates) == 0:
                file.write('===\n')
            else:
                for joint in skeltonCordinates:
                    file.write(', '.join([str(e) for e in joint])+'\n')

    def writeVideoFile(self, filePath, frameRate):
        writer = cv2.VideoWriter(filePath, cv2.VideoWriter_fourcc(
            *'XVID'), frameRate, (640, 480), True)
        for i in self.videos:
            writer.write(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))

    def writeBinaryDepthFile(self, filePath):
        np.array(self.depthImages).tofile(filePath)

    def addSkelton(self, ut, user):
        skeltonCordinates = []
        for idx in [nite2.JointType.NITE_JOINT_HEAD, nite2.JointType.NITE_JOINT_NECK,
                    nite2.JointType.NITE_JOINT_LEFT_SHOULDER,
                    nite2.JointType.NITE_JOINT_RIGHT_SHOULDER,
                    nite2.JointType.NITE_JOINT_LEFT_ELBOW,
                    nite2.JointType.NITE_JOINT_RIGHT_ELBOW,
                    nite2.JointType.NITE_JOINT_LEFT_HAND,
                    nite2.JointType.NITE_JOINT_RIGHT_HAND,
                    nite2.JointType.NITE_JOINT_TORSO,
                    nite2.JointType.NITE_JOINT_LEFT_HIP,
                    nite2.JointType.NITE_JOINT_RIGHT_HIP,
                    nite2.JointType.NITE_JOINT_LEFT_KNEE,
                    nite2.JointType.NITE_JOINT_RIGHT_KNEE,
                    nite2.JointType.NITE_JOINT_LEFT_FOOT,
                    nite2.JointType.NITE_JOINT_RIGHT_FOOT]:
            jnt = user.skeleton.joints[idx]
            (x1, y1) = ut.convert_joint_coordinates_to_depth(
                jnt.position.x, jnt.position.y, jnt.position.z)
            skeltonCordinates.append(
                [jnt.position.x, jnt.position.y, jnt.position.z, x1, y1])
        self.skeltons.append(skeltonCordinates)


class MicRecorder(object):

    def __init__(self, device, dirPath, channels=1):
        self.record = True
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self.run, args=())
        self.device = device
        self.dirPath = dirPath
        device_info = sd.query_devices(device, 'input')
        self.samplerate = int(device_info['default_samplerate'])
        self.channels = channels

    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status)
        self.q.put(indata.copy())

    def run(self):
        filename = self.dirPath+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + \
            "__"+str(self.device)+".wav"
        with sf.SoundFile(filename, mode='x', samplerate=self.samplerate,
                          channels=self.channels) as file:
            with sd.InputStream(samplerate=self.samplerate, device=self.device,
                                channels=self.channels, callback=self.callback):
                while self.record:
                    file.write(self.q.get())
        print("Finish audio recording at ",
              datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))

    def startTheThread(self):
        self.thread.start()

    def stopTheThread(self):
        self.record = False


def getFilePath(dirPath, subject, action):
    if not os.path.exists(dirPath+subject):
        os.mkdir(dirPath+subject)
    dirPath = dirPath+subject+"\\"
    if not os.path.exists(dirPath+action):
        os.mkdir(dirPath+action)
    return dirPath+action+"\\"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-a', '--action', type=str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-s', '--subject', type=str,
    help='subject')
parser.add_argument(
    '-d', '--device', type=str,
    help='subject')
parser.add_argument(
    '-p', '--path', type=str,
    help='location to save files')
args = parser.parse_args()


devices = [int(x) for x in args.device.split(",")]
mics = []
try:
    if (args.action == None) or (args.subject == None) or (args.device == None) or (args.path == None):
        parser.exit("Arguments missing!!")
    else:
        basePath = args.path
        dirPath = getFilePath(basePath, args.subject, args.action)
        myKinect = KinectRecoder(dirPath=dirPath)
        for device in devices:
            mics.append(MicRecorder(dirPath=dirPath, device=device))
        print("Starting devices at :",
              datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
        time.sleep(10)
        print("Recording Starting at :",
              datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
        myKinect.startTheThread()
        for mic in mics:
            mic.startTheThread()
        time.sleep(600)
        myKinect.stopTheThread()
        for mic in mics:
            mic.stopTheThread()
        print("Recording Stopping at :",
              datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
        time.sleep(20)
        parser.exit("Done")
except KeyboardInterrupt:
    myKinect.stopTheThread()
    for mic in mics:
        mic.stopTheThread()
    print("Recording Stopping at :",
          datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
    time.sleep(20)
    parser.exit("Done")
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))