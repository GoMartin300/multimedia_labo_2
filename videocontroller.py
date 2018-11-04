import cv2
import numpy as np
import tempfile
import os
from region import Region
from VideoWriter import VideoWriter
from spectralresidualsaliency import getSaliency
from fractions import Fraction

class VideoController(object):
    MINIMAL_HORIZONTAL_DIFFERENCE_FOR_MOVEMENT = 30
    MINIMAL_VERTICAL_DIFFERENCE_FOR_MOVEMENT = 30
    MOVE_HORIZONTAL_BETWEEN_FRAMES = 5
    MOVE_VERTICAL_BETWEEN_FRAMES = 3

    SKIP_ZOOM_CHANGE_BETWEEN_FRAMES = 4
    original_video_path = None
    cap = None
    orgtempdir = None
    targetheight = None
    targetwidth = None
    last_region = None

    sub_region_width = None
    sub_region_height = None


    def __init__(self, sys_argv):
        self.targetheight = int(sys_argv[2])
        self.targetwidth = int(sys_argv[3])
        self.sub_region_height = self.targetheight
        self.sub_region_width = self.targetwidth
        self.original_video_path = str(sys_argv[1])
        self.cap = cv2.VideoCapture(self.original_video_path)
        self.frame_amount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_ratio = None

        self.skipzoom = 0
    def calculate_threshold(self, saliency_image):
        # return cv2.adaptiveThreshold(saliency_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 2)
        # ret, thresh = cv2.threshold(saliency_image, np.mean(saliency_image), 255, cv2.THRESH_BINARY)
        ret, thresh = cv2.threshold(saliency_image, np.percentile(saliency_image, 80), 255, cv2.THRESH_BINARY)
        return thresh


    def calculate_bounding_rectangle(self, threshold_image):
        im, contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea)
        if contours.__len__() is not 0:
            rect = cv2.boundingRect((contours[-1]))
            return rect
        else:
            return [self.last_region.w1, self.last_region.h1, self.last_region.w2-self.last_region.w1, self.last_region.h2 - self.last_region.h1]

    # def shape_to_region(self, shape):
    #      return Region(w1=shape[0], h1=shape[1], w2=shape[0] + shape[2],
    #                      h2=shape[1] + shape[3])

    # def find_most_salient_region(self, sal_frame, original_height, original_width):
    #     sal_frame = cv2.resize(sal_frame, (128, 128))
    #     sal_height = sal_frame.shape[0]
    #     sal_width = sal_frame.shape[1]
    #     scaling_factor_h = original_height/sal_height
    #     scaling_factor_w = original_width/sal_width
    #     scaled_targetheight = round(self.targetheight//scaling_factor_h)
    #     scaled_targetwidth = round(self.targetwidth//scaling_factor_w)
    #     best_value = 0
    #     summed_area_matrix = np.cumsum(np.cumsum(sal_frame, axis=1, dtype='uint64'), axis=0, dtype='uint64')
    #     for h in range(0, sal_height - scaled_targetheight):
    #         for w in range(0, sal_width - scaled_targetwidth):
    #             A = summed_area_matrix[h][w]
    #             B = summed_area_matrix[h][w + scaled_targetwidth]
    #             C = summed_area_matrix[h + scaled_targetheight][w]
    #             D = summed_area_matrix[h + scaled_targetheight][w + scaled_targetwidth]
    #             value = D - B + A - C
    #             if best_value < value:
    #                 best_value = value
    #                 best_region = Region(h1=h, h2=h + scaled_targetheight, w1=w, w2=w + scaled_targetwidth)
    #     if best_value is 0:
    #         return self.last_region
    #     h1 = round(best_region.h1 * scaling_factor_h)
    #     w1 = round(best_region.w1 * scaling_factor_w)
    #     if h1 < 0:
    #         h1 = 0
    #     else:
    #         temp = h1 + self.targetheight - original_height
    #         if temp > 0:
    #             h1 = h1 - temp
    #     if w1 < 0:
    #         w1 = 0
    #     else:
    #         temp = w1 + self.targetwidth - original_width
    #         if temp > 0:
    #             w1 = w1 - temp
    #     return Region(w1, h1, w1 + self.targetwidth - 1, h1 + self.targetheight - 1)

    def fix_bounding_box_scale(self, i_box, org_frame_heigth, org_frame_width):
        self.original_ratio = Fraction(org_frame_width, org_frame_heigth)
        h1 = i_box[1]
        w1 = i_box[0]
        h_box = i_box[3]
        w_box = i_box[2]
        i_box_ratio = w_box/h_box
        if i_box_ratio == self.original_ratio.numerator/self.original_ratio.denominator:
            return i_box
        if i_box_ratio < (org_frame_width/org_frame_heigth):        #this means height is too big
            h_box_round = h_box//self.original_ratio.denominator * self.original_ratio.denominator

            if h_box_round < h_box:
                h_box_round = h_box_round + self.original_ratio.denominator
            w_round = (h_box_round * self.original_ratio.numerator)//self.original_ratio.denominator

            if h_box_round + h1 > org_frame_heigth:
                h_box_round = h_box_round - self.original_ratio.denominator
            h2 = h1 + h_box_round
            w_added = w_round - w_box
            w1_temp = w1 - w_added//2
            if w1_temp < 0:
                w1 = 0
                w_added = w_added - w1 - 1
                w2 = w1 + w_round
            else:
                w1 = w1_temp
                w2_temp = w1 + w_round
                if w2_temp >= org_frame_width:
                    excess = w2_temp - org_frame_width
                    w1 = w1 - excess
                w2 = w1 + w_round

        else:
            w_box_round = w_box//self.original_ratio.numerator * self.original_ratio.numerator

            if w_box_round < w_box:
                w_box_round = w_box_round + self.original_ratio.numerator
            h_round = (w_box_round * self.original_ratio.denominator)//self.original_ratio.numerator

            if w_box_round + w1 > org_frame_width:
                w_box_round = w_box_round - self.original_ratio.numerator
            w2 = w1 + w_box_round
            h_added = h_round - h_box
            h1_temp = h1 - h_added // 2
            if h1_temp < 0:
                h1 = 0
                h_added = h_added - h1 - 1
                h2 = h1 + h_round
            else:
                h1 = h1_temp
                h2_temp = h1 + h_round
                if h2_temp >= org_frame_heigth:
                    excess = h2_temp - org_frame_heigth
                    h1 = h1 - excess
                h2 = h1 + h_round

        return Region(w1 = w1, h1 = h1, w2=w2, h2 = h2)

    def cut_by_region(self, frame, region):
        return frame[region.h1:region.h2 + 1, region.w1:region.w2 + 1]

    """
    calculates the final frame that needs to be displayed, this means it makes sure the tranisition is smooth
    """
    def calculate_final_region(self, target_region, last_region, org_height, org_width):
        if last_region is None:
            return target_region
        temp = target_region.h1 - last_region.h1
        if temp > self.MINIMAL_VERTICAL_DIFFERENCE_FOR_MOVEMENT:
            h1 = last_region.h1 + self.MOVE_VERTICAL_BETWEEN_FRAMES
        elif abs(temp) > self.MINIMAL_VERTICAL_DIFFERENCE_FOR_MOVEMENT:
            h1 = last_region.h1 - self.MOVE_VERTICAL_BETWEEN_FRAMES
        else:
            h1 = last_region.h1
        temp = target_region.w1 - last_region.w1
        if temp > self.MINIMAL_HORIZONTAL_DIFFERENCE_FOR_MOVEMENT:
            w1 = last_region.w1 + self.MOVE_HORIZONTAL_BETWEEN_FRAMES
        elif abs(temp) > self.MINIMAL_HORIZONTAL_DIFFERENCE_FOR_MOVEMENT:
            w1 = last_region.w1 - self.MOVE_HORIZONTAL_BETWEEN_FRAMES
        else:
            w1 = last_region.w1

        w2 = w1 + self.sub_region_width
        h2 = h1 + self.sub_region_height
        self.skipzoom = self.skipzoom + 1
        if self.skipzoom >= self.SKIP_ZOOM_CHANGE_BETWEEN_FRAMES:
            self.skipzoom = 0

            last_region_width = last_region.w2 - last_region.w1
            last_region_height = last_region.h2 - last_region.h1
            if (target_region.w2 - target_region.w1) - last_region_width > self.original_ratio.numerator:
                if w2 + self.original_ratio.numerator < org_width:
                    w2 = w2 + self.original_ratio.numerator
                elif w1 - self.original_ratio.numerator > 0:
                    w1 = w1 - self.original_ratio.numerator

                if h2 + self.original_ratio.denominator < org_height:
                    h2 = h2 + self.original_ratio.denominator
                elif h1 - self.original_ratio.denominator > 0:
                    h1 = h1 - self.original_ratio.denominator
            elif last_region_width - (target_region.w2 - target_region.w1) > self.original_ratio.numerator:
                w2 = w2 - self.original_ratio.numerator
                h2 = h2 - self.original_ratio.denominator

            self.sub_region_height = h2 - h1
            self.sub_region_width = w2 - w1
        return Region(w1=w1, h1=h1, w2=w2, h2=h2)

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

    # def calculate_for_all(self):
    #     count = 0
    #     a = os.getcwd()
    #     framerate = self.cap.get(cv2.CAP_PROP_FPS)
    #     exp_video = cv2.VideoWriter('video.avi', cv2.VideoWriter.fourcc(*'MJPG'), framerate, (self.targetwidth, self.targetheight), False)
    #     last_region = None
    #     succes = True
    #     while succes:
    #         succes, frame = self.cap.read()
    #         if succes:
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #             # cv2.imshow('org', frame)
    #             sal_frame_small = getSaliency(frame)
    #             # cv2.imshow('sal', sal_frame_small)
    #             sal_frame_small = self.calculate_threshold(sal_frame_small)
    #
    #             cv2.imshow('thress', sal_frame_small)
    #             target_region = self.find_most_salient_region(sal_frame_small, frame.shape[0], frame.shape[1])
    #
    #             final_region = self.calculate_final_region(target_region, self.last_region)
    #             final_output_frame = self.cut_by_region(frame, final_region)
    #             self.last_region = final_region
    #             # cv2.imshow('final', self.cut_by_region(frame, final_region))
    #             exp_video.write(final_output_frame)
    #             # cv2.waitKey(1)
    #             # cv2.imwrite(os.path.join(self.saltempdir.name, str(count) + '.png'), final_output_frame)
    #             count += 1
    #             if count % 60 is 0:
    #                 print('finished second ' + str(count//60) + ' of ' + str(self.frame_amount//60))
    #
    #     # for picture in os.listdir(os.fsencode(self.orgtempdir.name)):
    #     exp_video.release()

    def calculate_for_all(self):
        count = 0
        a = os.getcwd()
        framerate = self.cap.get(cv2.CAP_PROP_FPS)
        exp_video = cv2.VideoWriter('video.avi', cv2.VideoWriter.fourcc(*'MJPG'), framerate, (self.targetwidth, self.targetheight), False)
        last_region = None
        succes = True
        while succes:
            succes, frame = self.cap.read()
            if succes:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sal_frame_small = getSaliency(frame)
                threshold_image = self.calculate_threshold(sal_frame_small)
                bounding_rect = self.calculate_bounding_rectangle(threshold_image)
                target_region = self.fix_bounding_box_scale(bounding_rect, frame.shape[0], frame.shape[1])
                final_region = self.calculate_final_region(target_region, self.last_region, frame.shape[0], frame.shape[1])
                final_output_frame = cv2.resize(self.cut_by_region(frame, final_region), (self.targetwidth, self.targetheight))
                self.last_region = final_region
                exp_video.write(final_output_frame)
                count += 1
                if count % 60 is 0:
                    print('finished second ' + str(count//60) + ' of ' + str(self.frame_amount//60))
        exp_video.release()


    def finish(self):
        self.cap.release()
