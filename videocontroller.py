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
    last_region = None

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
                count += 1

    # def calculate_saliency(self):
    #     count = 0
    #     for picture in os.listdir(self.orgtempdir):
    #         saliencymap = getSaliency(picture)
    #         cv2.imwrite(os.path.join(self.saltempdir.name, saliencymap, str(count) + '.jpg'))
    #         count += 1

    def calculate_threshold(self, saliency_image):
        return cv2.adaptiveThreshold(saliency_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)

    def find_most_salient_region(self, sal_frame):
        # amount_h_steps = 18
        # amount_w_steps = 32
        org_height = sal_frame.shape[0]
        org_width = sal_frame.shape[1]
        # h_step = (org_height-self.targetheight)//amount_h_steps
        # w_step = (org_width - self.targetwidth)//amount_w_steps
        best_value = 0b11111111111111111111111111111111
        summed_area_matrix = np.cumsum(np.cumsum(sal_frame, axis=1, dtype='uint64'), axis=0, dtype='uint64')
        region_list = []
        for h in range(0, org_height - self.targetheight):
            for w in range(0, org_width - self.targetwidth):
                A = summed_area_matrix[h][w]
                B = summed_area_matrix[h][w + self.targetwidth]
                C = summed_area_matrix[h + self.targetheight][w]
                D = summed_area_matrix[h + self.targetheight][w + self.targetwidth]
                value = D - B + A - C
                if best_value > value:
                    best_value = value
                    best_region = Region(h1=h, h2=h + self.targetheight, w1=w, w2=w + self.targetwidth)
        return best_region

    """
    calculates the final frame that needs to be displayed, this means it makes sure the tranisition is smooth
    """
    def calculate_next_frame(self, target_region, last_region):
        if last_region is None:
            return target_region
        else:

    # def get_region_blob(self,keypoints):
    #     im = cv2.drawKeypoints(keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    #     keypLeft = int
    #     keypRight = int
    #     keypTop = int
    #     keypBottom = int
    #
    #     #get most left,right,top,bottom pixel
    #     leftFound = False
    #     for x in range(0,len(im[0])):
    #         for y in range(0, len(im)):
    #             if im.item(y, x, 2)==255:
    #                 keypLeft = x
    #                 leftFound = True
    #                 break
    #         if leftFound:
    #             break
    #     rightFound = False
    #     for x in range(len(im[0]), 0):
    #         for y in range(0, len(im)):
    #             if im.item(y, x, 2)==255:
    #                 keypRight = x
    #                 rightFound = True
    #                 break
    #         if rightFound:
    #             break
    #     topFound = False
    #     for y in range(len(im), 0):
    #         for x in range(0, len(im[0])):
    #             if im.item(y, x, 2)==255:
    #                 keypTop = y
    #                 topFound = True
    #                 break
    #         if topFound:
    #             break
    #     bottomFound = False
    #     for y in range(0,len(im)):
    #         for x in range(0, len(im[0])):
    #             if im.item(y, x, 2)==255:
    #                 keypBottom = y
    #                 bottomFound = True
    #                 break
    #         if bottomFound:
    #             break
    #     return Region(keypLeft,keypBottom,keypRight,keypTop)

    # def blob_analysis(self, sal_frame): # https://www.learnopencv.com/blob-detection-using-opencv-python-c/
        # # set up the detector
        # detector = cv2.SimpleBlobDetector()
        # # detect blobs
        # keypoints = detector.detect(sal_frame)
        #
        # # test function
        # # draw blobs as circles
        # im_with_keypoints = cv2.drawKeypoints(sal_frame, keypoints, np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow("keyPoints", im_with_keypoints)
        # cv2.waitKey(0)
        # return self.get_region_blob(keypoints)

    def calculate_for_all(self):
        count = 0
        a = os.getcwd()
        # big_video = VideoWriter(os.path.join(os.getcwd(), 'multimedia_labo_3'), 'big', 1080, 1920)
        # small_video = VideoWriter(os.path.join(os.getcwd(), 'multimedia_labo_3'), 'small', self.targetheight, self.targetwidth)
        last_region = None
        for picture in os.listdir(os.fsencode(self.orgtempdir.name)):
            frame = cv2.imread(os.path.join(self.orgtempdir.name, str(count) + '.jpg'), 0)
            frame = self.calculate_threshold(frame)
            target_region = self.find_most_salient_region(frame)
            final_frame = self.calculate_next_frame(target_region, last_region)
            # big_video.addframe(frame)
            # cv2.imwrite(os.path.join(self.saltempdir.name, str(count) + '.jpg'), frame)
            # self.blob_analysis(frame)
            # region = self.find_most_salient_region(frame)
            # small_video.addframe(frame[region.h1:region.h2+1, region.w1:region.w2+1])
            last_region = target_region
            count += 1
        # big_video.finish()
        # small_video.finish()

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

    # def prefered_region(self):
        # When you crop the image to a preferred width & height
