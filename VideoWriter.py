import cv2
import os

class VideoWriter(object):
    videodirectory = None
    cv2videowriter = None

    def __init__(self, directory, videoname, targetheight=None, targetwidth=None):
        if targetwidth is None or targetheight is None:
            self.cv2videowriter = cv2.VideoWriter(os.path.join(directory, videoname, '.mjpg'),
                                                  cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'), 60,
                                                  cv2.CV_WINDOWS_AUTOSIZE, 0)

        self.cv2videowriter = cv2.VideoWriter(os.path.join(directory, videoname, 'mjpg'), cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'), 60, (targetheight, targetwidth), 0)

    def addframe(self, frame):
        self.cv2videowriter.write(frame)

    def add_directory_frames(self, directory):
        count = 0
        for picture in os.listdir(os.fsencode(directory)):
            self.cv2videowriter(cv2.imread(os.path.join(directory.name, str(count) + '.jpg')))
            count += 1

    def finish(self):
        self.cv2videowriter.release()




