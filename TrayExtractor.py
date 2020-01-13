import os
import pickle

import cv2
import numpy

import DataStructures

TRAY_ID = 128
TRAY_TYPE = 1
try:
    os.mkdir("/Users/gghosal/Documents/PipelineOutputTray/" + str(TRAY_ID) + "_" + str(TRAY_TYPE))
    # print
except:
    pass


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


def breakdown_filename(filename: str):
    split_by_underscore = filename.split("_")
    date = int(split_by_underscore[0])
    time = int(split_by_underscore[2])
    return (date, time)


for err, shelf_file in [('/Users/gghosal/Desktop/gaurav_new_photos/Errors31/corrections_dict',
                         "/Users/gghosal/Desktop/gaurav_new_photos/ShelfFiles31"),
                        ('/Users/gghosal/Desktop/gaurav_new_photos/Errors32/corrections_dict',
                         "/Users/gghosal/Desktop/gaurav_new_photos/ShelfFiles32"),
                        ('/Users/gghosal/Desktop/gaurav_new_photos/Errors41/corrections_dict',
                         "/Users/gghosal/Desktop/gaurav_new_photos/ShelfFiles41"),
                        ('/Users/gghosal/Desktop/gaurav_new_photos/Errors42/corrections_dict',
                         "/Users/gghosal/Desktop/gaurav_new_photos/ShelfFiles42"),
                        ('/Users/gghosal/Desktop/gaurav_new_photos/Errors51/corrections_dict',
                         "/Users/gghosal/Desktop/gaurav_new_photos/ShelfFiles51"),
                        ('/Users/gghosal/Desktop/gaurav_new_photos/Errors52/corrections_dict',
                         "/Users/gghosal/Desktop/gaurav_new_photos/ShelfFiles52"),
                        ('/Users/gghosal/Desktop/gaurav_new_photos/Errors61/corrections_dict',
                         "/Users/gghosal/Desktop/gaurav_new_photos/ShelfFiles61"),
                        ('/Users/gghosal/Desktop/gaurav_new_photos/Errors62/corrections_dict',
                         "/Users/gghosal/Desktop/gaurav_new_photos/ShelfFiles62")
                        ]:
    corrections = pickle.load(open(err, "rb"))
    for i in listdir_nohidden(shelf_file):
        os.chdir(shelf_file)
        filename_no_path = os.path.split(i)[-1]
        filename_no_extension = filename_no_path.split(".")[0]
        data = breakdown_filename(filename_no_extension)
        trays_list = pickle.load(open(i, "rb"))
        for tray in trays_list:
            tray.tray_id = corrections.get(str(tray.tray_id), str(tray.tray_id))
            if not str(tray.tray_id).isalpha():
                try:
                    tray.tray_id = int(tray.tray_id)
                except:
                    continue
            if tray.tray_id == TRAY_ID and (tray.orientation == DataStructures.Tray.orientation1).all():
                # pot=tray.get_pot_position(POT_POSITION)
                grid = numpy.array(tray.get_all_pots()).reshape((2, 4), order='F')
                # for a,b in zip(grid[0],grid[1]):
                # print(a.pot_position, b.pot_position)
                img = numpy.hstack([numpy.vstack([a.get_image(), b.get_image()]) for a, b in zip(grid[0], grid[1])])
                # print(img)
                # cv2.imshow('ho',img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # print(img)
                cv2.imwrite(
                    "/Users/gghosal/Documents/PipelineOutputTray/" + str(TRAY_ID) + "_" + str(TRAY_TYPE) + "/" + str(
                        data[0]) + "_" + str(data[1]) + ".jpg", img)
