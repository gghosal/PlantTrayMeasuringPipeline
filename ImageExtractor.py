import os
import pickle

import cv2

# 3L3008_1.png
# _11.png
BARCODE = '/Users/gghosal/Downloads/rep_3_barcode.csv'
barcode = open(BARCODE, 'rt')
barcode_lines = barcode.readlines()
barcode_lines = [i.rstrip() for i in barcode_lines]
barcode_dict = dict([(i.split(",")[1], i.split(",")[0]) for i in barcode_lines])
print(barcode_dict)
TRAY_ID = 69  # barcode_dict['71']#int(sys.argv[1])
print(TRAY_ID)
POT_POSITION = 1  # int(sys.argv[2

try:
    # shutil.rmtree("/Users/gghosal/Documents/PipelineOutput/Temp/")
    os.mkdir("/Users/gghosal/Documents/PipelineOutput/Temp/""/" + str(TRAY_ID) + "_" + str(POT_POSITION) + "/")
except:
    os.chdir("/Users/gghosal/Documents/PipelineOutput/Temp/""/" + str(TRAY_ID) + "_" + str(POT_POSITION) + "/")


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
    print(shelf_file)
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
            if tray.tray_id == TRAY_ID:
                pot = tray.get_pot_position(POT_POSITION)
                # print(tray.get_all_pots())
                try:
                    img = pot.get_image()
                except:
                    continue
                cv2.imwrite("/Users/gghosal/Documents/PipelineOutput/" + str("Temp") + "/" + str(TRAY_ID) + "_" + str(
                    POT_POSITION) + "/" + str(data[0]) + "_" + str(data[1]) + ".jpg", img)
