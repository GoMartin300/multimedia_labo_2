import cv2
import os


class VideoWriter(object):
    videodirectory = None
    cv2videowriter = None

    def __init__(self, directory, videoname, targetheight, targetwidth):
        self.cv2videowriter = cv2.VideoWriter(os.path.join(directory, videoname, 'mjpg'), 60, (targetheight, targetwidth), 0)

    def addframe(self, frame):
        self.cv2videowriter.write(frame)

    # def add_directory_frames(self, directory):
    #     count = 0
    #     for picture in os.listdir(os.fsencode(directory)):
    #         self.cv2videowriter.write(cv2.imread(os.path.join(directory.name, str(count) + '.img')))
    #         count += 1

    def finish(self):
        self.cv2videowriter.release()




