#####Executable Pipeline


import ImageProcUtil
import HalfShelfSegmenter
import NoiseRemoval
import HalfPotSegmenter
import DotCodeReader

class ExecutablePipeline:
    def __init__(self):
        self.shelf_segmenter=HalfShelfSegmenter.HalfShelfSegmenter('/Users/gghosal/Desktop/ProgramFilesPlantPipeline/Vertical.jpg','/Users/gghosal/Desktop/Template.jpg',1400,500)
        self.potsegmenter=HalfPotSegmenter.HalfPotSegmenter()
        self.dotcodereader=DotCodeReader.DotCodeReader("/Users/gghosal/Desktop/dotcodestranslated.dat",{'red':0, "lightblue":98, "darkblue":120, "pink":173, "purple":160})
        
