import time
import re
import os
import cv2
import numpy as np
import nidaqmx  
from pycromanager import Core

class Camera_snap:  
    def __init__(self):
        self.doPort = None
        self.exposure = 100  # ms
        self.core = None

    def initialize_camera(self):
        # Initialise Camera, daq card
        self.doPort = nidaqmx.Task()
        self.core = Core() 
        doPortName = "Dev1/port0/line2"
        self.doPort.do_channels.add_do_chan(doPortName)

        self.core.set_exposure(self.exposure) 
        self.core.set_property('pco_camera', 'Acquiremode', 'External')  
        self.core.set_property('pco_camera', 'Triggermode', 'External')  

        self.core.initialize_circular_buffer()

    def snap(self):
        self.doPort.write(True)  # make sure camera has stopped by requesting a final unused image
        self.doPort.write(False)

    def capture_image_and_save(self, position):
        if self.core.is_sequence_running():
            self.core.stop_sequence_acquisition()  # stop the camera
            self.snap()  # Take an image on the camera

        self.core.start_continuous_sequence_acquisition(0)  # start the camera

        self.snap()  # Take an image on the camera

        while self.core.get_remaining_image_count() == 0:  # wait until picture is available
            time.sleep(0.001)
        result = self.core.pop_next_tagged_image()

        # Save image
        self.snap()
        # set interval for collecting imgs, needed to avoid motion blur while the stage is moving during the autofocus search
        time.sleep(0.1)
        result = self.core.pop_next_tagged_image()
        # reshape if needed
        pixels = np.squeeze(np.reshape(result.pix, newshape=[-1, result.tags["Height"], result.tags["Width"]],)) 
        pixels = pixels.astype('float64')  

        # Save image
        filename = f"image_position_{position}"
        save_path = os.path.join(self.path, filename + '.tif')
        # Convert the image to 8-bit depth
        pixels_8bit = cv2.normalize(pixels.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Save the image
        cv2.imwrite(save_path, pixels_8bit)
        print(pixels.shape)
        print(f"Image saved to: {save_path}")

        # Stop the camera, and the last pics won't be saved
        self.snap()
        if self.core.is_sequence_running():
            print('Still running')
            self.core.stop_sequence_acquisition()
        self.snap()

        # Exit camera
        self.doPort.close()

        print("Finished")

        return pixels_8bit
