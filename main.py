from videocontroller import VideoController

import sys

a = VideoController(sys.argv)
a.calculate_for_all()
a.finish()
