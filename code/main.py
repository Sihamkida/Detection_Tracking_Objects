

import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
from main_scripts import calibration
#from main_scripts import classify
#from main_scripts import object_tracking

if __name__  == '__main__':
    #Start calibration
    calibration.readImgs("../../Calibration/calibration_img")
    