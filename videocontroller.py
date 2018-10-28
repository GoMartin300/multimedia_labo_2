import cv2
import numpy as np
import tempfile
import os
from region import Region
from VideoWriter import VideoWriter
from spectralresidualsaliency import getSaliency


class VideoController(object):
    original_video_path = None
    cap = None
    orgtempdir = None
    targetheight = None
    targetwidth = None

    def __init__(self, sys_argv):
        self.targetheight = int(sys_argv[2])
        self.targetwidth = int(sys_argv[3])
        self.orgtempdir = tempfile.TemporaryDirectory(prefix='orgvideoframes', dir=os.getcwd())
        self.saltempdir = tempfile.TemporaryDirectory(prefix='salvideoframes', dir=os.getcwd())

        self.original_video_path = str(sys_argv[1])
        self.cap = cv2.VideoCapture(self.original_video_path)
        # self.frame_amount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        succes = True
        count = 0
        while succes:
            succes, frame = self.cap.read()
            if succes:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(self.orgtempdir.name, str(count) + '.jpg'), frame)
                # cv2.imwrite(os.path.join(self.saltempdir.name, str(count) + '.jpg'), codefromjakob(frame))
                count += 1

    def calculate_saliency(self):
        count = 0
        for picture in os.listdir(self.orgtempdir):
            saliencymap = getSaliency(picture)
            cv2.imwrite(os.path.join(self.saltempdir.name,saliencymap,str(count) + '.jpg'))
            count += 1

    def calculate_binary(self, saliency_image):
        return cv2.adaptiveThreshold(saliency_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)

    def find_region(self, sal_frame):
        amount_h_steps = 18
        amount_w_steps = 32
        org_height = sal_frame.shape[0]
        org_width = sal_frame.shape[1]
        h_step = (org_height-self.targetheight)//amount_h_steps
        w_step = (org_width - self.targetwidth)//amount_w_steps
        region_list = []
        for h in range(0, amount_h_steps + 1):
            for w in range(0, amount_w_steps + 1):
                temp_region = Region()
                temp_region.h1 = h * h_step
                temp_region.h2 = temp_region.h1 + self.targetheight
                temp_region.w1 = w * w_step
                temp_region.w2 = temp_region.w1 + self.targetwidth
                region_list.append(temp_region)

        max = 0
        max_region_number = 0
        for region_number in range(0, len(region_list)):
            region = region_list[region_number]
            cut_frame = sal_frame[region.h1:region.h2+1, region.w1:region.w2+1]
            region_saliency_sum = np.sum(cut_frame)
            if region_saliency_sum > max:
                max = region_saliency_sum
                max_region_number = region_number
        return region_list[max_region_number]


    def calculate_for_all(self):
        count = 0
        a = os.getcwd()
        big_video = VideoWriter(os.path.join(os.getcwd(), 'multimedia_labo_3'), 'big', 1080, 1920)
        small_video = VideoWriter(os.path.join(os.getcwd(), 'multimedia_labo_3'), 'small', self.targetheight, self.targetwidth)
        for picture in os.listdir(os.fsencode(self.orgtempdir.name)):
            frame = cv2.imread(os.path.join(self.orgtempdir.name, str(count) + '.jpg'), 0)
            frame = self.calculate_binary(frame)
            big_video.addframe(frame)
            cv2.imwrite(os.path.join(self.saltempdir.name, str(count) + '.jpg'), frame)
            blob_analysis(self,frame)
            region = self.find_region(frame)
            small_video.addframe(frame[region.h1:region.h2+1, region.w1:region.w2+1])
            count += 1
        big_video.finish()
        small_video.finish()

        # videowriter =
        # for picture in os.listdir(os.fsencode(self.orgtempdir.name)):
        #     frame = cv2.imread(os.path.join(self.saltempdir.name, str(count) + '.jpg'))
        #
        #     cv2.imshow('big', frame)
        #     cv2.imshow('small', frame[region.h1:region.h2+1, region.w1:region.w2+1])
        #     cv2.waitKey(1)
        #     count += 1


    def finish(self):
        self.cap.release()


    def blob_analysis(self, sal_frame): # https://www.learnopencv.com/blob-detection-using-opencv-python-c/
        # set up the detector
        detector = cv2.SimpleBlobDetector()
        # detect blobs
        keypoints = detector.detect(sal_frame)

        # test function
        # draw blobs as circles
        im_with_keypoints = cv2.drawKeypoints(sal_frame, keypoints, np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("keyPoints", im_with_keypoints)
        cv2.waitKey(0)

    def get_region(self):

    def prefered_region(self):

        # When you crop the image to a preferred width & height
