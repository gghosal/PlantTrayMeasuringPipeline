######Datastructures.py
"""Contains data structure implementations which are made use of in the process of storing and tracking the measurements for the different plants.
This module is imported by others and the two classes are used to manage and store the data."""
import numpy as np
class Pot:
    """Contains a single pot image"""
    def __init__(self, tray, pot_position, image):
        self.tray_id=tray
        self.pot_position=pot_position
        self.image=image
        self.measurement=0
    def output_identifier_csv(self):
        return str(self.tray_id)+","+str(self.pot_position)+","+str(self.measurement)
    def store_measurement(self, measurement):
        self.measurement=measurement
    def get_image(self):
        return self.image
class Tray:
    orientation1=np.array([[1,2],[5,6],[9,10],[13,14]])
    orientation2=np.array([[3,4],[7,8],[11,12],[15,16]])
    def __init__(self, tray_id, orientation):
        """Orientation is either 1 or 2 and refers to the tray identifiers which are present"""
        self.potlist=list()
        if orientation==1:
            self.orientation=np.array([[1,2],[5,6],[9,10],[13,14]])
        elif orientation==2:
            self.orientation=np.array([[3,4],[7,8],[11,12],[15,16]])
        self.tray_id=tray_id
    def scan_in_pots(self, pots):
        orientation=self.orientation.flatten(order="C")
        potsorder=list(zip(orientation,pots))
        for i in potsorder:
            self.potlist.append(Pot(self.tray_id,i[0],i[1]))
    def get_pot_position(self, position_index):
        for i in self.potlist:
            if i.pot_position==position_index:
                return i
        return None
    def get_all_pots(self):
        return self.potlist

    
    
    
            
            
        
