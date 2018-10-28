from videocontroller import VideoController
import sys

videoController = VideoController(sys.argv)
videoController.calculate_for_all()
videoController.finish()
