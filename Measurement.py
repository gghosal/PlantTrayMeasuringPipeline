####MeasurementSystem
import os.path
import pickle

import cv2
import numpy as np
from plantcv import plantcv as pcv

import MeasurementSaver

"""Takes in a directory of ".shelf" files and an error-correction file and """

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


class MeasurementSystem:
    """Contains executable code for generating measurments from .shelf files"""

    def __init__(self, measurements_to_create):
        """measurements_to_create refers to a list of measurements to create"""
        self.measurements = measurements_to_create

    def breakdown_filename(self, filename: str):
        split_by_underscore = filename.split("_")
        date = int(split_by_underscore[0])
        time = int(split_by_underscore[2])
        return (date, time)

    def threshold_green(self, image):
        # image=cv2.convertScaleAbs(image,image, 1.25,0)
        # cla=cv2.createCLAHE()#sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # image=cv2.filter2D(image, -1,sharpen_kernel)
        # image=cla.apply(image)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        device = 0

        avg = np.mean(img_hsv[:, :, 2])
        img_hsv[:, :, 2] = cv2.add((90 - avg), img_hsv[:, :, 2])
        # print("hi")
        green_lower = np.array([30, 100, 60])  ##Define lower bound found by experimentation
        green_upper = np.array([90, 253, 255])  ##Upper bound
        mask = cv2.inRange(img_hsv, green_lower, green_upper)
        device, dilated = pcv.dilate(mask, 1, 1, device)
        device, mask = pcv.fill(dilated, dilated, 30, device)
        # device, dilated = pcv.fill(dilated, dilated, 50, device)
        res = cv2.bitwise_and(img_hsv, img_hsv, mask=dilated)
        # plt.imshow(res)
        # plt.show()
        # cv2.imshow("hi",res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return dilated, res

    def threshold_green_old(self, image):
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        device = 0
        # print("hi")
        green_lower = np.array([30, 120, 120])  ##Define lower bound found by experimentation
        green_upper = np.array([90, 253, 255])  ##Upper bound
        mask = cv2.inRange(img_hsv, green_lower, green_upper)
        #device, mask=pcv.fill(mask, mask, 50, device)
        device, dilated = pcv.dilate(mask, 1, 1, device)
        #device, dilated = pcv.fill(dilated, dilated, 50, device)
        res = cv2.bitwise_and(img_hsv, img_hsv, mask=dilated)
        # plt.imshow(res)
        # plt.show()
        # cv2.imshow("hi",res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return dilated, res



    def process_pot(self, pot_image):
        device = 0
        # debug=None
        updated_pot_image = self.threshold_green(pot_image)
        # plt.imshow(updated_pot_image)
        # plt.show()
        device, a = pcv.rgb2gray_lab(updated_pot_image, 'a', device)
        device, img_binary = pcv.binary_threshold(a, 127, 255, 'dark', device, None)
        # plt.imshow(img_binary)
        # plt.show()
        mask = np.copy(img_binary)
        device, fill_image = pcv.fill(img_binary, mask, 50, device)
        device, dilated = pcv.dilate(fill_image, 1, 1, device)
        device, id_objects, obj_hierarchy = pcv.find_objects(updated_pot_image, updated_pot_image, device)
        device, roi1, roi_hierarchy = pcv.define_roi(updated_pot_image, 'rectangle', device, None, 'default', debug,
                                                     False)
        device, roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(updated_pot_image, 'partial', roi1,
                                                                               roi_hierarchy, id_objects, obj_hierarchy,
                                                                               device, debug)
        device, obj, mask = pcv.object_composition(updated_pot_image, roi_objects, hierarchy3, device, debug)
        device, shape_header, shape_data, shape_img = pcv.analyze_object(updated_pot_image, "Example1", obj, mask,
                                                                         device, debug, False)
        print(shape_data[1])

    def process_pot2(self, pot_image):
        device = 0
        debug = None
        update_pot_image, res = self.threshold_green(pot_image)
        # plt.imshow(update_pot_image)
        # plt.show()
        # device, a=pcv.rgb2gray_lab(update_pot_image, 'a', device)
        # device, img_binary=pcv.binary_threshold(a, 1, 255, 'dark', device, None)
        # plt.imshow(img_binary)
        # plt.show()
        contours, w2 = cv2.findContours(update_pot_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        area = 0

        for i in contours:
            area += cv2.contourArea(i)
            # cv2.drawContours(img_binary, i, -1, (random.randint(0,255), random.randint(0,255), random.randint(0,255)),1)
        # plt.imshow(img_binary)
        # plt.show()
        return area

    def process_pot2_old(self, pot_image):
        device = 0
        debug = None
        update_pot_image, res = self.threshold_green_old(pot_image)
        # plt.imshow(update_pot_image)
        # plt.show()
        # device, a=pcv.rgb2gray_lab(update_pot_image, 'a', device)
        # device, img_binary=pcv.binary_threshold(a, 1, 255, 'dark', device, None)
        # plt.imshow(img_binary)
        # plt.show()
        contours, w2 = cv2.findContours(update_pot_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        area = 0

        for i in contours:
            area += cv2.contourArea(i)
            # cv2.drawContours(img_binary, i, -1, (random.randint(0,255), random.randint(0,255), random.randint(0,255)),1)
        # plt.imshow(img_binary)
        # plt.show()
        return area


if __name__ == '__main__':

    a = MeasurementSystem("hi")
    corrections = pickle.load(open('/Users/gghosal/Desktop/gaurav_new_photos/Errors31/corrections_dict', "rb"))
    print(corrections)
    with MeasurementSaver.MeasurementSaver(
            database_path='/Users/gghosal/Desktop/gaurav_new_photos/measurements16.db') as saver:
        saver.create_database('/Users/gghosal/Desktop/gaurav_new_photos/measurements10.db')

        for i in listdir_nohidden("/Users/gghosal/Desktop/gaurav_new_photos/ShelfFiles31"):
            # i="20131026_Shelf3_0600_1_masked.shelf"
            os.chdir("/Users/gghosal/Desktop/gaurav_new_photos/ShelfFiles31")
            filename_no_path = os.path.split(i)[-1]
            filename_no_extension = filename_no_path.split(".")[0]
            data = a.breakdown_filename(filename_no_extension)
            trays_list = pickle.load(open(i, "rb"))
            # print(trays_list)
            #print(corrections)
            # segmenter=HalfPotSegmenter.HalfPotSegmenter()
            # pots=segmenter.split_half_trays(r)
            for tray in trays_list:
                # print(tray.tray_id)
                # if str(tray.tray_id).isalpha():
                #print(corrections.get(str(tray.tray_id), str(tray.tray_id)))
                # if not str(corrections.get(str(tray.tray_id), str(tray.tray_id))).isalpha():
                tray.tray_id = corrections.get(str(tray.tray_id), str(tray.tray_id))

                if not str(tray.tray_id).isalpha():
                    try:
                        tray.tray_id = int(tray.tray_id)
                    except:
                        continue

                    for pot in tray.get_all_pots():
                        # cv2.imshow("hi",pot.get_image())
                        pot.tray_id = tray.tray_id
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        pot.store_measurement(a.process_pot2(pot.get_image()))
                        # print(pot.output_id_identifier_csv())
                        saver.save_singular_measurement(int(data[0]), int(data[1]), pot.tray_id, pot.pot_position,
                                                        pot.measurement)
