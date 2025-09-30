import argparse
import os
import random
import threading
import qi
import torch as torch
import cv2
import sys
import time

from PIL import Image
from models.experimental import attempt_load
from utils.torch_utils import time_sync
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy
from utils.plots import Annotator, colors

# IDs0
AL_kTopCamera = 0
AL_kBottomCamera = 1
AL_kDepthCamera = 2

# resolution image
AL_kQVGA = 1  # 320*240px
AL_kVGA = 2  # 640*480px

# Camera
camera_width = 640
camera_height = 480
###############################
# Settings needed from yolo
name = 'exp'  # save results to project/name
exist_ok = False  # existing project/name ok, do not increment
save_txt = False
visualize = False
device = 'cpu'  # device used for the recognition
half = False
imgsz = 640  # size of the image
augment = False  # augmented inference
classes = None  # filter by class: --class 0, or --class 0 2 3
conf_thres = 0.5  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 1000  # maximum detections per image
agnostic_nms = False  # class-agnostic NMS
save_crop = False  # save cropped prediction boxes
line_thickness = 3  # bounding box thickness (pixels)
view_img = True
save_conf = False  # save confidences in --save-txt labels
save_img = True
hide_labels = False,  # hide labels
hide_conf = False  # hide confidences
pt = True
model_path = "./yolov5s.pt"
object_names = ["bottle"]  # object we are searching for.
angle_X = 0.46
angle_Y = 0.20
radius = 20
vocal_thresh = 0.1


class CamDetection(object):
    def __init__(self, app):
        super(CamDetection, self).__init__()
        app.start()
        session = app.session

        # Service subscribed for this application
        self.videoRecorder = session.service("ALVideoRecorder")
        self.nav = session.service("ALNavigation")
        self.ba = session.service('ALBasicAwareness')
        self.aLife = session.service("ALAutonomousLife")
        self.tts = session.service("ALTextToSpeech")
        self.video_service = session.service("ALVideoDevice")
        self.motion_service = session.service("ALMotion")
        self.memory_service = session.service("ALMemory")
        self.sonar_service = session.service("ALSonar")
        self.tracker = session.service("ALTracker")
        self.robot_posture = session.service("ALRobotPosture")
        self.tabletService = session.service("ALTabletService")
        self.asr = session.service("ALSpeechRecognition")
        self.aLife.setAutonomousAbilityEnabled("All", True)
        # Subscribe to sonars, this will launch sonars (at hardware level)
        # and start data acquisition.
        self.sonar_service.subscribe("app")

        # Object attributes
        self.objectFound = False  # True if Pepper found the object, False anyway
        self.seen_count = 0  # Count of how many times has Pepper seen the object before identifying it
        self.isExploring = False  # True if Pepper is exploring the area, False anyway
        self.seenObject = False  # True if Pepper have seen the object during the Exploration
        self.isChecking = False  # True if Pepper is checking the Object, False anyway

        # Object coordinates
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.centerX = 0
        self.centerY = 0

    def run(self):

        self.asr.pause(True)
        self.asr.setLanguage("English")
        self.asr.setLanguage("Italian")
        self.asr.removeAllContext()
        self.memory_service.insertData("WordRecognized", ["null", 0])

        # Example: Adds "yes", "no" and "please" to the vocabulary (without wordspotting)
        vocabulary = ["trova", "bottiglia"]
        self.asr.setVocabulary(vocabulary, False)

        # Start the speech recognition engine with user Test_ASR
        self.asr.pause(False)
        self.asr.subscribe("Test_ASR")

        print("Speech recognition engine started")

        while True:

            word = self.memory_service.getData("WordRecognized")
            x, y = word[0], word[1]
            print(word)
            if str(x) in vocabulary and word[1] >= vocal_thresh:
                print(x)
                print(y)
                self.memory_service.removeData("WordRecognized")
                break

        self.asr.unsubscribe("Test_ASR")
        print("Speech recognition engine stopped")
        self.search_init()


    def search_init(self):
        # self.ba.setEnabled(False)  # Disabling the Basic Awareness
        self.isExploring = True
        threading.Thread(target=self.nav.explore, args=(radius,)).start()

        # Wait some seconds before starting the Object Recognition and giving Pepper enough time to start the exploration
        time.sleep(7)
        self.startSearching()

        self.nav.stopExploration()
        cv2.destroyAllWindows()
        self.ba.setEnabled(True)

    def startSearching(self):
        model = attempt_load(model_path, map_location=device)
        stride = int(model.stride.max())  # model stride
        strName = "imageCapturing{}".format(random.randint(1, 10000000000))
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names

        # subscribeCamera((name of the module), (Camera Index), (Resolution), (Color Space), (Fps))
        clientRGB = self.video_service.subscribeCamera(strName, AL_kTopCamera, AL_kVGA, 11, 10)

        while True:
            imageRGB = self.video_service.getImageRemote(clientRGB)
            width = imageRGB[0]
            height = imageRGB[1]
            array = imageRGB[6]
            image_string = bytes(bytearray(array))
            # Create a PIL Image from our pixel array.
            img = Image.frombytes("RGB", (width, height), image_string)

            dataset = LoadImages(img, img_size=imgsz, stride=stride, auto=".pt")
            dt, seen = [0.0, 0.0, 0.0], 0

            for img, im0s in dataset:
                t1 = time_sync()
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(img.shape) == 3:
                    img = img[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1

                # Inference
                if pt:
                    pred = model(img, augment=augment, visualize=visualize)[0]

                t3 = time_sync()
                dt[1] += t3 - t2

                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3
                processed = False

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    s, im0 = '', im0s.copy()
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        """
                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        """

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_img or save_crop or view_img:  # Add bbox to image
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                c = int(cls)  # integer class
                                label = f'{names[c]} {conf:.2f}'
                                # print('Label: ' + label + '\nxywh: ' + str(xywh))
                                annotator.box_label(xyxy, label, color=colors(c, True))
                                print(names[int(cls)])

                                # if the searched object is seen in the video stream, save the coordinates and the size
                                # of the object and start the checking algorithm
                                if names[int(cls)] in object_names:
                                    processed = True
                                    self.x = xywh[0]  # x coordinate
                                    self.y = xywh[1]  # y coordinate
                                    self.w = xywh[2]  # width of the object
                                    self.h = xywh[3]  # height of the object
                                    self.centerX = self.x + (self.w / 2)  # center x coordinate
                                    self.centerY = self.y + (self.h / 2) - 0.1  # center y coordinate
                                    self.seenObject = True
                                    if not self.isChecking:
                                        threading.Thread(target=self.objectRecognition).start()

                    if not processed:
                        self.seenObject = False

                    print(self.seenObject)

                    # Print time (inference-only)
                    print(f'{s}Done. ({t3 - t2:.3f}s)')

                    # Stream results
                    im0 = annotator.result()
                    if view_img:
                        cv2.imshow("image", im0)
                        cv2.waitKey(1)  # 1 millisecond

        cap.release()
        cv2.destroyAllWindows()

    def objectRecognition(self):
        # this algorithm checks if the object is still in  Pepper's field of view
        # if not, move Pepper's head to the sides and check again
        # at the end, if the object is not detected, Pepper restarts the exploration
        self.isChecking = True  # start the checking phase of the object

        # if Pepper is still exploring, stop the exploration and the movements
        if self.isExploring:
            self.nav.stopExploration()
            self.motion_service.stopMove()
            self.isExploring = False
            self.aLife.setAutonomousAbilityEnabled("All", False)


        time.sleep(6)
        if self.seenObject:
            self.centerObject()
        else:
            self.moveHead(0, -1, False)
            time.sleep(5)
            if self.seenObject:
                self.centerObject()
            else:
                self.moveHead(0, 0, False)
                time.sleep(5)
                if self.seenObject:
                    self.centerObject()
                else:
                    self.moveHead(0, 1, False)
                    time.sleep(5)
                    if self.seenObject:
                        self.centerObject()
                    else:
                        self.isChecking = False
                        threading.Thread(target=self.nav.explore, args=(radius,)).start()
                        self.isExploring=True


    def centerObject(self):
        # if the object is detected, Pepper centers his head and body towards it
        # angleX and angleY are multiplicative constants used to obtain the movement to be made with
        # the head from the center to the object

        if self.centerX < 0.5:
            mov = 0.5 - self.centerX
            self.moveHead(0, mov, True)
            print("moving left: " + str(mov))
        elif self.centerX > 0.5:
            mov = (angle_X * (self.centerX - 0.5)) / 0.5
            self.moveHead(0, -mov, True)
            print("moving right: " + str(mov))

        time.sleep(1)

        if self.centerY < 0.5:
            mov = 0.5 - self.centerY
            self.moveHead(-mov, 0, True)
            print("moving up: " + str(mov))
        elif self.centerY > 0.5:
            mov = (angle_Y * (self.centerY - 0.5)) / 0.5
            self.moveHead(mov, 0, True)
            print("moving down: " + str(mov))

        time.sleep(2)
        self.centerBody()

    def moveHead(self, amntY, amntX, amnt):
        # HeadPitch :{(-)up,(+)down} , HeadYaw :{(+)right,(-)left} (angolo di arrivo)
        jointNames = ["HeadPitch",
                      "HeadYaw"]
        headPitch_angle, headYaw_angle = self.motion_service.getAngles(jointNames, False)
        fractionMaxSpeed = 0.09

        if amnt:
            HeadA = [float(headPitch_angle + amntY), float(headYaw_angle + amntX)]
        else:
            HeadA = [float(amntY), float(amntX)]

        self.motion_service.setAngles(jointNames, HeadA, fractionMaxSpeed)
        print("moving head started")

    def centerBody(self):
        # the body is centered by rotating it at the same angle as the head
        # after, the head is centered with the body

        JointNames = ["HeadPitch",
                      "HeadYaw"]
        fractionMaxSpeed = 0.09
        headPitch_angle, headYaw_angle = self.motion_service.getAngles(JointNames, False)
        self.motion_service.moveTo(0, 0, headYaw_angle)
        time.sleep(1)
        self.motion_service.setAngles("HeadYaw", 0, fractionMaxSpeed)
        time.sleep(2)

        # self.motion_service.move(0.1, 0, 0)
        # Pepper begins to move slowly until he reaches the first obstacle he encounters
        self.moveArm("LArm", [1, 0, 0], 0, 0.1)
        self.tts.say("Li c'è una bottiglia")

        time.sleep(2)
        self.robot_posture.goToPosture("StandInit", 1.0)

    def moveArm(self, effector, position, frame, fracMaxSpeed):
        # Effector – effector name. Could be “Arms”, “LArm”, “RArm”.
        # Position – position 3D [x, y, z].
        # Frame – position frame {FRAME_TORSO = 0, FRAME_WORLD = 1, FRAME_ROBOT = 2}.
        # FractionMaxSpeed – a fraction.
        self.tracker.pointAt(effector, position, frame, fracMaxSpeed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="pepper.local",
                        help="Robot IP address. On robot or Local Naoqi: use '192.168.2.2'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    try:
        # Initialize qi framework.
        connection_url = "tcp://" + args.ip + ":" + str(args.port)
        app = qi.Application(["CamDetection", "--qi-url=" + connection_url])
    except RuntimeError:
        print("Can't connect to Naoqi at ip \"" + args.ip + "\" on port "
              + str(args.port) + ".\n"
                                 "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    cam_detection = CamDetection(app)
    cam_detection.run()
