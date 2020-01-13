import os
import subprocess
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

import Measurement


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


INPUT_FILE = '/Users/gghosal/Downloads/r3_high_std_err_plants.csv'
BARCODE = '/Users/gghosal/Downloads/rep_3_barcode.csv'
input_file = open(INPUT_FILE, "rt")
barcode = open(BARCODE, 'rt')
a = Measurement.MeasurementSystem([])
barcode_lines = barcode.readlines()
barcode_lines = [i.rstrip() for i in barcode_lines]
barcode_dict = dict([(i.split(",")[1], i.split(",")[0]) for i in barcode_lines])
input_lines = input_file.readlines()[1:]
input_lines = [i.rstrip() for i in input_lines]
for title in reversed(input_lines):
    # if title+".png" in os.lis18tdir("/Users/gghosal/Documents/PipelineOutput/NewMeasurementsGraph/"):
    # print(title)
    # continue
    tray, pot = title.split("_")[0], title.split("_")[1]
    tray_number = barcode_dict[tray]
    subprocess.call(
        [sys.executable, "/Users/gghosal/Documents/GitHub/PlantTrayMeasuringPipeline/ImageExtractor.py", tray_number,
         pot])
    measurements_old = list()
    measurements_new = list()
    for i in listdir_nohidden("/Users/gghosal/Documents/PipelineOutput/" + str("Temp") + "/"):
        os.chdir("/Users/gghosal/Documents/PipelineOutput/" + str("Temp") + "/")
        img = cv2.imread(i)
        measurements_old.append(a.process_pot2_old(img))
        measurements_new.append(a.process_pot2(img))

        # print("done")
    measurements_old = list(map(lambda x: np.log(x), measurements_old))
    measurements_new = list(map(lambda x: np.log(x), measurements_new))
    fig, ax = plt.subplots()
    # plt.yscale("log")

    ax.scatter(range(len(measurements_old)), measurements_old, label="old_threshold")
    ax.scatter(range(len(measurements_new)), measurements_new, label="new_threshold")
    fig.suptitle(title)
    ax.legend()
    # ax.axis('scaled')
    # plt.show()
    plt.savefig("/Users/gghosal/Documents/PipelineOutput/NewMeasurementsGraph/" + str(title) + ".png",
                bbox_inches='tight')
    plt.close()
