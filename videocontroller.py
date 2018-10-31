import cv2
import numpy as np
import tempfile
import os
from region import Region
from VideoWriter import VideoWriter
from spectralresidualsaliency import getSaliency


class VideoController(object):
    MOVE_HORIZONTAL_BETWEEN_FRAMES = 1
    MOVE_VERTICAL_BETWEEN_FRAMES = 1

    original_video_path = None
    cap = None
    orgtempdir = None
    targetheight = None
    targetwidth = None
    last_region = None

    def __init__(self, sys_argv):
        self.targetheight = int(sys_argv[3])
        self.targetwidth = int(sys_argv[2])

        self.original_video_path = str(sys_argv[1])
        self.cap = cv2.VideoCapture(self.original_video_path)
        self.frame_amount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def calculate_threshold(self, saliency_image):
        # return cv2.adaptiveThreshold(saliency_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 2)
        ret, thresh = cv2.threshold(saliency_image, np.mean(saliency_image), 255, cv2.THRESH_BINARY)
        return thresh

    def find_most_salient_region_temp(self, sal_frame):
        org_height = sal_frame.shape[0]
        org_width = sal_frame.shape[1]
        best_value = 0
        summed_area_matrix = np.cumsum(np.cumsum(sal_frame, axis=1, dtype='uint64'), axis=0, dtype='uint64')
        for h in range(0, org_height - self.targetheight):
            for w in range(0, org_width - self.targetwidth):
                A = summed_area_matrix[h][w]
                B = summed_area_matrix[h][w + self.targetwidth]
                C = summed_area_matrix[h + self.targetheight][w]
                D = summed_area_matrix[h + self.targetheight][w + self.targetwidth]
                value = D - B + A - C
                if best_value < value:
                    best_value = value
                    best_region = Region(h1=h, h2=h + self.targetheight, w1=w, w2=w + self.targetwidth)
        return best_region

    def find_most_salient_region(self, sal_frame, original_height, original_width):
        sal_height = sal_frame.shape[0]
        sal_width = sal_frame.shape[1]
        scaling_factor_h = original_height/sal_height
        scaling_factor_w = original_width/sal_width
        scaled_targetheight = round(self.targetheight//scaling_factor_h)
        scaled_targetwidth = round(self.targetwidth//scaling_factor_w)
        best_value = 0
        summed_area_matrix = np.cumsum(np.cumsum(sal_frame, axis=1, dtype='uint64'), axis=0, dtype='uint64')
        for h in range(0, sal_height - scaled_targetheight):
            for w in range(0, sal_width - scaled_targetwidth):
                A = summed_area_matrix[h][w]
                B = summed_area_matrix[h][w + scaled_targetwidth]
                C = summed_area_matrix[h + scaled_targetheight][w]
                D = summed_area_matrix[h + scaled_targetheight][w + scaled_targetwidth]
                value = D - B + A - C
                if best_value < value:
                    best_value = value
                    best_region = Region(h1=h, h2=h + scaled_targetheight, w1=w, w2=w + scaled_targetwidth)
        try:
            h1 = round(best_region.h1 * scaling_factor_h)
        except UnboundLocalError:
            a = 2+2
        w1 = round(best_region.w1 * scaling_factor_w)
        if h1 < 0:
            h1 = 0
        else:
            temp = h1 + self.targetheight - original_height
            if temp > 0:
                h1 = h1 - temp
        if w1 < 0:
            w1 = 0
        else:
            temp = w1 + self.targetwidth - original_width
            if temp > 0:
                w1 = w1 - temp
        return Region(w1, h1, w1 + self.targetwidth - 1, h1 + self.targetheight - 1)

    def cut_by_region(self, frame, region):
        return frame[region.h1:region.h2 + 1, region.w1:region.w2 + 1]

    """
    calculates the final frame that needs to be displayed, this means it makes sure the tranisition is smooth
    """
    def calculate_final_region(self, target_region, last_region):
        if last_region is None:
            return target_region

        temp = target_region.h1 - last_region.h1
        if temp > self.MOVE_VERTICAL_BETWEEN_FRAMES:
            h1 = last_region.h1 + self.MOVE_VERTICAL_BETWEEN_FRAMES
        elif abs(temp) > self.MOVE_VERTICAL_BETWEEN_FRAMES:
            h1 = last_region.h1 - self.MOVE_VERTICAL_BETWEEN_FRAMES
        else:
            h1 = last_region.h1
        temp = target_region.w1 - last_region.w1
        if temp > self.MOVE_HORIZONTAL_BETWEEN_FRAMES:
            w1 = last_region.w1 + self.MOVE_HORIZONTAL_BETWEEN_FRAMES
        elif abs(temp) > self.MOVE_HORIZONTAL_BETWEEN_FRAMES:
            w1 = last_region.w1 - self.MOVE_HORIZONTAL_BETWEEN_FRAMES
        else:
            w1 = last_region.w1
        return Region(w1=w1, h1=h1, w2=w1 + self.targetwidth - 1, h2=h1 + self.targetheight - 1)

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
        exp_video = cv2.VideoWriter('video.avi', cv2.VideoWriter.fourcc(*'MJPG'), 60, (self.targetheight, self.targetwidth), False)
        # small_video = VideoWriter(os.path.join(os.getcwd(), 'multimedia_labo_3'), 'small', self.targetheight, self.targetwidth)
        last_region = None
        succes = True
        while succes:
            succes, frame = self.cap.read()
            if succes:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sal_frame_small = getSaliency(frame)
                sal_frame_small = self.calculate_threshold(sal_frame_small)
                target_region = self.find_most_salient_region(sal_frame_small, frame.shape[0], frame.shape[1])

                final_region = self.calculate_final_region(target_region, last_region)
                final_output_frame = self.cut_by_region(frame, final_region)
                last_region = final_region
                # cv2.imshow('final', self.cut_by_region(frame, final_region))
                # small_video.addframe(final_output_frame)
                exp_video.write(final_output_frame)
                # cv2.waitKey(1)
                # cv2.imwrite(os.path.join(self.saltempdir.name, str(count) + '.png'), final_output_frame)
                count += 1
                if count % 60 is 0:
                    print('finished second ' + str(count//60) + ' of ' + str(self.frame_amount//60))
            # big_video.addframe(frame)
            # self.blob_analysis(frame)
            # region = self.find_most_salient_region(frame)
            # small_video.addframe(frame[region.h1:region.h2+1, region.w1:region.w2+1])

        # for picture in os.listdir(os.fsencode(self.orgtempdir.name)):
        exp_video.release()
        # small_video.finish()

    def finish(self):
        self.cap.release()

    # def prefered_region(self):
        # When you crop the image to a preferred width & height
