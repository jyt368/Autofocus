import serial
import cv2
import numpy as np
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial
from scipy.optimize import fmin
from scipy.optimize import least_squares
# import matplotlib.pyplot as plt
import time
import re
import os
import nidaqmx  # microscope control
from pycromanager import Core
# from pycromanager import Bridge
import tifffile
from tkinter.messagebox import showinfo
from ctypes import c_char_p
import glob
import math
from skimage import io
import matplotlib.pyplot as plt
# import torch
import argparse
import datetime
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import gaussian
from scipy.optimize import minimize
from NIQE1 import niqe
#from sewar.full_ref import niqe
from skimage import filters, color
from scipy.ndimage import generic_filter


class StageController:
    def __init__(self, path='Z:/Users/yj368/Autofocus_repeat/Autofocus_repeat/TIRF/5'):
        self.path=path
        self.core = Core()

    def connect_to_stage(self):
        print('Connecting to Stage Control')
        try:
            # Assuming the device label for your stage in Micro-Manager is 'XYStage' and 'ZStage'
            self.core.set_focus_device('ZStage')
            print('Connection to Stage Control Complete')
            return True
        except Exception as e:
            print('Exception: Connecting to stage: ' + str(e))
            return False
    
    def get_stage_position(self):
        try:
            # Get the current Z position of the stage, self.core.get_position()
            position_response = self.core.get_position()
            #print(position_response)
            current_position = float(position_response)
            print('current_position=',current_position)
            print(f'Set Z position to {position_response} um')
            return current_position

        except Exception as e:
            print('Exception: Setting Z position: ' + str(e))
            return None

    def parse_response_for_position(self, response):
        # This function would parse the response string and return the Z position.
        # This is just an example, the actual parsing will depend on the format of the response.
        match = re.search(r':A (-?\d+)', response) # find the first occurrence of :A followed by one or more digits
        if match: # if a match is found
            z_position = match.group(1) # get the matched substring, which is the number after :A
            return float(z_position) # convert it to a float and return it
        else: # if no match is found
            return None # return None
    
    def capture_image_and_save(self, position, exposure=100):
        self.core.set_exposure(exposure)

        if self.core.is_sequence_running(): 
            self.core.stop_sequence_acquisition() # stop the camera
            self.core.snap_image() #Take an image on the camera

        self.core.snap_image()  # Take an image on the camera
        time.sleep(0.01)
        print('image taken')

        #result = self.core.get_last_image()  # Get the image data
        result = self.core.get_tagged_image()
        
        # reshape if needed
        pixels = np.squeeze(np.reshape(result.pix,newshape=[-1, result.tags["Height"], result.tags["Width"]],))
        
        #pixels = np.squeeze(np.reshape(result, newshape=[-1, result.shape[0], result.shape[1]]))  # reshape image data

        filename = f"image_position_{position}"
        save_path = os.path.join(self.path, filename + '.tif')
        # Save the image
        tifffile.imwrite(save_path, pixels)
        print(f"Image saved to: {save_path}")

        print("Finished")

        return pixels
    
    def capture_image_and_save1(self, position, exposure=100):
        #self.core.set_property('Camera','Binning','2x2')
        #self.core.set_property('Camera','Channel','mcherry')


        if self.core.is_sequence_running(): 
            self.core.stop_sequence_acquisition() # stop the camera
            self.core.snap_image() #Take an image on the camera

        self.core.set_exposure(exposure) # ref 9
        # core.set_property('pco_camera','Acquiremode','External') # ref 9
        # core.set_property('pco_camera','Triggermode','External') # ref 9
        # core.set_property('Camera-1','AcquisitionMethod','Pooling')
        # core.set_property('Camera-1','TriggerMode','Edge Trigger')

        self.core.initialize_circular_buffer()
        self.core.start_continuous_sequence_acquisition(0) # start the camera

        self.core.snap_image() #Take an image on the camera
        print('image taken')

        while self.core.get_remaining_image_count() == 0: #wait until picture is available
            # time.sleep(0.001)
            time.sleep(0.001)
        result = self.core.pop_next_tagged_image()

        # Save image
        self.core.snap_image()
        # set interval for collecting imgs
        time.sleep(0.01)
        result = self.core.pop_next_tagged_image() 
        # reshape if needed
        pixels = np.squeeze(np.reshape(result.pix,newshape=[-1, result.tags["Height"], result.tags["Width"]],)) # reshape image data


        filename = f"image_position_{position}"
        save_path = os.path.join(self.path, filename + '.tif')
        # Save the image
        tifffile.imwrite(save_path, pixels)
        print(f"Image saved to: {save_path}")

        #Stop the camera, and the last pics won't be saved
        self.core.snap_image()
        if self.core.is_sequence_running():
            print('Still running')
            self.core.stop_sequence_acquisition() 
        self.core.snap_image()


        print("Finished")
        
        return pixels
    
    def move_stage(self, pos):
        z_axis_value = float(pos)    #unit micron, stage movement unit: micron
        self.core.set_position(z_axis_value)
        print(f"Moved Z stage to {z_axis_value}")
        # time.sleep(2)  # Give some time for the stage to stabilize

    def calculate_snr(self, image):
        # Convert image to grayscale if it's a color image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean and standard deviation of the image
        mean, std_dev = cv2.meanStdDev(image)
        
        # Calculate signal and noise
        signal = mean[0]
        noise = std_dev[0]
        
        # Calculate SNR
        snr = signal / noise
        
        return snr[0]
    
    def denoise_image(self, image, kernel_size=1, sigma=1):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Step 1: Neighborhood Averaging (Box Filter)
        # image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        neighborhood_averaging = cv2.boxFilter(image.astype(np.uint8), -1, (3, 3), normalize=True)
        # Step 2: Bilateral Filtering for Edge Preservation
        bilateral_denoised_image = cv2.bilateralFilter(image.astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75)
        # Combine the results using a weighted sum for better noise reduction
        combined_denoised_image = cv2.addWeighted(neighborhood_averaging, 0.5, bilateral_denoised_image, 0.5, 0)
        # Step 3: Apply Gaussian filter for additional smoothing
        denoised_image = cv2.GaussianBlur(combined_denoised_image, (kernel_size, kernel_size), sigma)
        # Display the clean image
        '''plt.imshow(denoised_image, cmap='gray')
        plt.axis('off')
        plt.title('Clean Image')
        plt.show()'''
        return denoised_image
    
    def process_image(self, observed_image, subtraction_value=0, saturation_factor= 0.035):
        # Ensure both images are of the same size
        # observed_image=self.denoise_image(observed_image)
        # observed_image = cv2.resize(observed_image, (self.background_image.shape[1], self.background_image.shape[0]))

        # get the shape of the background image
        # height, width= self.background_image.shape

        # resize the observed image to the same shape
        # observed_image = cv2.resize(observed_image, (width, height))

        # now you can subtract them
        # subtracted_image = cv2.subtract(observed_image, self.background_image*2)
        subtracted_image = cv2.subtract(observed_image, subtraction_value)
    
        # Enhance contrast with saturated pixels
        enhanced_image = exposure.adjust_sigmoid(subtracted_image, gain=5, cutoff=saturation_factor)
    
        # Normalize the image
        # normalized_image = cv2.normalize(enhanced_image, None, 0, 255, cv2.NORM_MINMAX)
        '''plt.imshow(enhanced_image, cmap='gray')
        plt.axis('off')
        plt.title('Clean Image')
        plt.show()'''

    
        return enhanced_image

    def calculate_sharpness(self,image):
        
        # Step 2: Laplace Gradient Operator
        laplacian_image = cv2.Laplacian(image, cv2.CV_64F)
        laplacian_image_8bit = cv2.convertScaleAbs(laplacian_image)

        # Step 3: Local Variance Information (FGLOG)
        global_average_value = np.mean(laplacian_image)
        rows, cols = laplacian_image.shape
        fglog = np.sum((laplacian_image- global_average_value)**2) / (rows * cols)
        print('evaluation value:',fglog)

        return fglog
    
    def calculate_sharpness1(self,image):
        # Step 1: Convert to grayscale if not already
        if image.ndim == 3:
            image_gray = color.rgb2gray(image)
        else:
            image_gray = image

        # Step 2: Calculate the gradient magnitude using Sobel operator
        gradient = np.sqrt(filters.sobel_h(image_gray)**2 + filters.sobel_v(image_gray)**2)
        sharpness_metric_sobel = np.mean(gradient)
        print('sharpness_metric_sobel:', sharpness_metric_sobel)

        # Step 3: Calculate Laplacian
        laplacian_image = cv2.Laplacian(image_gray, cv2.CV_64F)
        global_average_value = np.mean(laplacian_image)
        rows, cols = laplacian_image.shape
        fglog = np.sum((laplacian_image - global_average_value)**2) / (rows * cols)

        # Step 4: Combine or use the metrics as needed
        # For example, you can return the average of the two metrics
        std_sobel = np.std(gradient)
        std_fglog = np.std(laplacian_image - global_average_value)
        sharpness_metric_sobel_scaled = sharpness_metric_sobel / std_sobel
        fglog_scaled = fglog / std_fglog
        combined_sharpness = (sharpness_metric_sobel_scaled + fglog_scaled) / 2

        return combined_sharpness
    

    def defocus_hill_search(self, step_size=1, threshold=120, abs_threshold_slope=0.03):
        current_position = self.get_stage_position()
        current_step_size = step_size
        consecutive_count = 0
        num = 0
        count=0
        cons_num=0
        prev_sharp=0
        prev_slope=None
        peak=0
        max_sharp=0
        small=0
        prev_sharpness, snr, pos, slopes = [], [], [], [] # Initialize with a large value to avoid stopping prematurely
        
        current_image=self.capture_image_and_save(current_position)
        mean = np.mean(current_image)
        threshold = mean
        print('std:', np.std(current_image))
        print('threshold:', threshold)

        for _ in range(5):
            image = self.capture_image_and_save(current_position)
            denoised_image = cv2.subtract(image, threshold)
            sharpness = self.calculate_sharpness(denoised_image)
            pos.append(current_position)
            prev_sharpness.append(sharpness)

            current_position =  current_position +current_step_size
            self.move_stage(current_position)
            prev_sharp = sharpness
            count+=1

        
        coefficients = np.polyfit(pos[-5:], prev_sharpness[-5:], 1)
        polynomial = np.poly1d(coefficients)
        derivative = polynomial.deriv()
        slope = derivative(current_position)
        '''x_new = np.linspace(pos[-5], pos[-1], num=5*10)
        plt.scatter(pos[-5:], prev_sharpness[-5:], label='Original Data')
        plt.plot(x_new, polynomial(x_new))
        plt.xlabel("Defocus distance (um)")
        plt.ylabel("Evaluation value (sharpness)")
        plt.legend()
        plt.show()'''

        direction = 1 if slope > 0 else -1

        print(direction)
        if direction==-1:
            prev_sharpness, pos, slopes=[],[],[]

        # Start the hill climbing search
        while True:
            count+=1
        
            current_position += direction * current_step_size
            self.move_stage(current_position)

            current_image = self.capture_image_and_save(current_position)
            denoised_image = cv2.subtract(current_image, threshold)
            #snr_value = self.calculate_snr(denoised_image)

            # Calculate sharpness
            image_sharpness = self.calculate_sharpness(denoised_image)
            print('sharpness', image_sharpness)
            pos.append(current_position)
            prev_sharpness.append(image_sharpness)

            if image_sharpness>max_sharp:
                max_sharp=image_sharpness
                peak= current_position

            curve, optimal_coef, derivative_curve = fit_quadratic_curve(pos[-10:], prev_sharpness[-10:])
            slope = derivative_curve[len(derivative_curve) - 1]
            print('slope:', slope)
            slopes.append(slope)
            '''extended_pos = np.linspace(min(pos[-10:]), max(pos[-10:]), 500)

            # Predict the second-order curve
            predicted_curve = np.dot(legendre_polynomials(extended_pos, 2), optimal_coef)

            # Move to the focus position
            poly = Polynomial(optimal_coef)
            print('coef:', poly)
            peak_position = -poly.coef[1] / (2 * poly.coef[2])
            print('peak_position:', peak_position)

            plt.scatter(pos[-10:], prev_sharpness[-10:], label='Original Data')
            plt.plot(extended_pos, predicted_curve, color='red')
            plt.xlabel("Defocus distance (um)")
            plt.ylabel("Evaluation value (sharpness)")
            plt.legend()
            plt.show()'''
            
            # adaptive slope threshold based on mean and standard deviation of the last 3 slopes
            avg_slope = np.mean(slopes[-3:])
            std_dev_slope = np.std(slopes[-3:])
                        
            print('avg_slope:', avg_slope)
            print('avg_slope - std_dev_slope:', avg_slope + std_dev_slope)
            if avg_slope<0:
                threshold_slope=abs(avg_slope + std_dev_slope)
            else:
                threshold_slope=abs(avg_slope - std_dev_slope)
            # Set an absolute threshold
            abs_threshold_slope = abs_threshold_slope*step_size*1e3  #unit nm


            # If the slope is small, switch to a smaller step size, 68% within 1 std, 95% within 2std, 99.7% within 3std.
            if abs(slope)<= threshold_slope and abs(slope)<abs_threshold_slope and num<2: 
                cons_num +=1
                print('cons_num:', cons_num)
                if cons_num >3 and num==0: 
                    current_step_size /= 2
                    num+=1
                    cons_num=0
                # Can change the step size to a even smaller value, large medium small
                if cons_num >3 and num==1: 
                    current_step_size /= 2
                    num+=1
            else:
                cons_num=0

            # If the slope has changed direction, stop the search
            if image_sharpness <= prev_sharp:
                consecutive_count += 1
                print('consecutive_count:', consecutive_count)
                if consecutive_count > 3:
                    break
            else:
                consecutive_count = 0

            prev_sharp = image_sharpness
            prev_slope = slope

        print('count:',count)
        if count >=10:
            pos=pos[-10:]
            prev_sharpness= prev_sharpness[-10:]

        return current_position, prev_sharpness, pos, peak


    def disconnect(self):
        try:
            #self.core.unload_all_devices()
            self.core.set_focus_device('ZStage')
        except Exception as e:
            print('Exception: Closing serial port: ' + str(e))
            

# Function for least-squares curve fitting using orthogonal polynomial functions
# Define Legendre polynomials
def denormalize(x_normalized, x_min, x_max):
    """Denormalize x from the [-1, 1] range back to its original scale."""
    return (x_normalized + 1) / 2 * (x_max - x_min) + x_min

def legendre_polynomials(x, order):
    return np.polynomial.legendre.legval(x, np.eye(order + 1)).T

# Define the error function for least-squares optimization
def error_function(coefficients, x, y, order):
    fitted_curve = np.dot(legendre_polynomials(x, order), coefficients)
    residuals = y - fitted_curve
    return residuals

def fit_quadratic_curve(x, y, order=2):
    x_shifted = x  # Shift x data so that the first value is 0
    derivative_basis_functions=[]
    # Initial guess for coefficients
    initial_guess =  np.ones(order + 1)
    #print(x)

    # Fit the quadratic curve
    result = least_squares(error_function, initial_guess, args=(x_shifted, y, order))

    # Extract fitted coefficients
    fitted_coefficients = result.x

    # Analyze the fitted curve for accuracy
    fitted_curve = np.dot(legendre_polynomials(x_shifted, order), fitted_coefficients)
    residuals = y - fitted_curve
    mean_squared_error = np.mean(residuals**2)
    print('Error:', mean_squared_error)

    for i in range(1, order + 1):
        derivative_basis_functions.append(i * np.array(x_shifted)**(i - 1))
    derivative_basis_functions = np.vstack(derivative_basis_functions).T
    
    # Calculate the derivative of the fitted curve
    derivative_curve = np.dot(derivative_basis_functions, fitted_coefficients[1:])
    print(fitted_coefficients[1:])

    '''plt.plot(x_shifted, derivative_curve, color='green', label='Derivative Curve')

    plt.xlabel("Defocus distance (um)")
    plt.ylabel("Evaluation value (sharpness)")
    plt.legend()
    plt.show()'''
    
    return fitted_curve, fitted_coefficients, derivative_curve

# Iterative curve fitting
def iterative_curve_fitting(x, y, max_order, accuracy_threshold=1e-10):
    x_shifted = x  # Shift x data so that the first value is 0
    order = 1  # Initial order
    coefficients = np.ones(order + 1)
    prev_error = np.inf

    while order <= max_order:
        result = least_squares(error_function, coefficients, args=(x_shifted, y, order))
        fitted_coefficients = result.x
        fitted_curve = np.dot(legendre_polynomials(x_shifted, order), fitted_coefficients)

        # Analyze the fitted curve for accuracy
        residuals = y - fitted_curve
        mean_squared_error = np.mean(residuals**2)
        print('Error:', abs(mean_squared_error))

        if abs(mean_squared_error) < accuracy_threshold or abs(mean_squared_error) >= abs(prev_error):
            order -=1
            coefficients = np.ones(order+1)
            result = least_squares(error_function, coefficients, args=(x_shifted, y, order))
            fitted_coefficients = result.x
            fitted_curve = np.dot(legendre_polynomials(x_shifted, order), fitted_coefficients)
            break  # If accuracy requirement is met or error starts increasing, exit the loop
        else:
            prev_error = mean_squared_error
            order += 1  # Increase the order for the next iteration
            coefficients = np.ones(order + 1)

    return fitted_curve, fitted_coefficients, order

def find_poly_max(poly):
    # Define the negative of the polynomial function
    neg_poly = lambda x: -poly(x)

    # Use scipy's fmin function to find the maximum
    max_x = fmin(neg_poly, 0)

    return max_x[0]

def normalise_to_minus_one_to_one(x):
    x_min, x_max = np.min(x), np.max(x)
    x_normalized = 2 * (x - x_min) / (x_max - x_min) - 1
    return x_normalized, x_min, x_max

def denormalise_from_minus_one_to_one(x_normalized, x_min, x_max):
    x_original = (x_normalized + 1) / 2 * (x_max - x_min) + x_min
    return x_original

def autofocus(focused_position, sharpness_values, pos, cs_threshold, max_poly_order=2):
    '''pos_array = np.array(pos)
    length= int(len(pos_array)/2)
    pos_shifted = (pos_array - int(pos_array[length]))*1000'''
    pos_shifted, x_min, x_max = normalise_to_minus_one_to_one(pos)
    best_fit_curve, best_fit_coefficients, best_fit_order = iterative_curve_fitting(pos_shifted, sharpness_values, max_poly_order)

    # Plot the results using original pos values
    plt.scatter(pos, sharpness_values, label='Original Data')
    plt.plot(pos, best_fit_curve, label=f'Fitted Curve (Order {best_fit_order})', color='red')
    plt.xlabel("Defocus distance (um)")
    plt.ylabel("Evaluation value (sharpness)")
    plt.legend()
    plt.savefig('C:/Users/Micro_Admin/anaconda3/test/search img/test/hill1.png')
    # plt.show()

    # Focused position is where the curve reaches its maximum
    poly = Polynomial(best_fit_coefficients)
    focused_position_sharp_normalized = find_poly_max(poly)
    focused_position_sharp = (focused_position_sharp_normalized + 1) / 2 * (x_max - x_min) + x_min
    print(focused_position_sharp)

    return focused_position_sharp


def normalize(x):
    x_min, x_max = np.min(x), np.max(x)
    x_normalized = (x - x_min) / (x_max - x_min)
    return x_normalized

def calculate_niqe(image):
    # Convert the image to grayscale
    if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the NIQE score
    niqe_score = niqe(image)

    return niqe_score

# Test:
if __name__ == '__main__':
    asi_controller = StageController()
    t=0
    asi_controller.disconnect()
    if asi_controller.connect_to_stage():
        # test move_stage--correct
        #asi_controller.move_stage(2000)  #unit nm
        #asi_controller.get_stage_position()
        start=time.time()
        focused_position, sharpness_values, pos, peak =asi_controller.defocus_hill_search()
        print('peak:', peak)
        
        # reference_image = cv2.imread("D:/jyt_dataset/defocus0.jpg", cv2.IMREAD_GRAYSCALE)
        focused_position_sharp= autofocus(focused_position, sharpness_values, pos, cs_threshold=0.9975)

        '''if abs(focused_position_sharp)>1e4 or focused_position_sharp==0:
            focused_position_sharp=peak
            t=1'''

        print(f"The Sharpness focused position is {focused_position_sharp} nm.")
        # print(f"The Sharpness focused position is {focused_position_cs} nm.")
        print("-----cost time:{:.4f}s----".format(time.time()-start))

        if t==0:
            asi_controller.move_stage(focused_position_sharp)
            time.sleep(0.2)
            image1= asi_controller.capture_image_and_save(focused_position_sharp, 100)
            niqe_score1 = calculate_niqe(image1)
            time.sleep(0.1)
            asi_controller.move_stage(peak)
            time.sleep(0.2)
            image2= asi_controller.capture_image_and_save(peak, 100)
            time.sleep(0.1)
            niqe_score2 = calculate_niqe(image2)

            if niqe_score1>niqe_score2:
                focused_position=peak
                print('The focused image is sharper')
            else:
                focused_position=focused_position_sharp
                print('The focused image is blurrier')
        else:
            focused_position=focused_position_sharp
            time.sleep(0.2)
            image2= asi_controller.capture_image_and_save(focused_position_sharp, 100)

        asi_controller.move_stage(focused_position)

        print(f"The Sharpness focused position is {focused_position} nm.")
        # print(f"The Sharpness focused position is {focused_position_cs} nm.")
        print("-----cost time:{:.4f}s----".format(time.time()-start))

        asi_controller.disconnect()
    else:
        print('Failed to connect to ASI Stage Controller')



