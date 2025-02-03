import serial
import cv2
import numpy as np
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial
from scipy.optimize import fmin
from scipy.optimize import least_squares
import time
import re
import os
import nidaqmx  # microscope control
from pycromanager import Core
# from pycromanager import Bridge
import tifffile
from tkinter.messagebox import showinfo
from ctypes import c_char_p
from skimage import io
import matplotlib.pyplot as plt
import argparse
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import gaussian
from scipy.optimize import minimize
from NIQE1 import niqe
from skimage import filters, color
from scipy.ndimage import generic_filter


class StageController:
    def __init__(self, path='.../Autofocus_repeat/time/100ms/11'):
        self.path=path
        self.core = Core()

    def connect_to_stage(self):
        print('Initializing Stage Connection...')
        try:
            # Check the 'Zstage' device name via micro-manager 
            self.core.set_focus_device('FocusDrive')
            print('Successfully connected to the stage')
            return True
        except Exception as e:
            print('Exception: Connecting to stage: ' + str(e))
            return False
    
    def get_stage_position(self):
        # Get the current Z position of the stage
        position_response = self.core.get_position()
        current_position = float(position_response)
        print('current_position=',current_position)
        print(f'Set Z position to {position_response} um')
        return current_position
    
    # Read the current position of the stage from the saved image file
    def parse_response_for_position(self, response):
        match = re.search(r':A (-?\d+)', response) 
        if match: 
            z_position = match.group(1) # get the matched first group string
            return float(z_position) 
        else: 
            return None 
    
    
    def capture_image_and_save(self, position, exposure=100):
        #self.core.set_property('Camera','Binning','2x2')
        #self.core.set_property('Camera','Channel','mcherry')
        self.core.set_exposure(exposure)

        if self.core.is_sequence_running(): 
            self.core.stop_sequence_acquisition() # stop the camera
            self.core.snap_image() 

        time.sleep(0.01)
        self.core.snap_image()  # Take an image on the camera
        print('image taken')

        result = self.core.get_tagged_image()
        
        # reshape if needed
        pixels = np.squeeze(np.reshape(result.pix,newshape=[-1, result.tags["Height"], result.tags["Width"]],))

        filename = f"image_position_{position}"
        save_path = os.path.join(self.path, filename + '.tif')
        # Save the image
        tifffile.imwrite(save_path, pixels)
        print(f"Image saved to: {save_path}")

        print("Finished")

        return pixels
    
    def move_stage(self, pos):
        z_axis_value = float(pos)    #unit micron, stage minimal movement unit: 0.01 micron, 10nm
        self.core.set_position(z_axis_value)
        print(f"Moved Z stage to {z_axis_value}")

    def calculate_snr(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        mean, std_dev = cv2.meanStdDev(image)
        
        # Calculate signal and noise
        signal = mean[0]
        noise = std_dev[0]
        
        # Calculate SNR
        snr = signal / noise
        
        return snr[0]

    def calculate_sharpness(self,image):
        
        # Edges: Laplace Operator
        laplacian_image = cv2.Laplacian(image, cv2.CV_64F)
        laplacian_image_8bit = cv2.convertScaleAbs(laplacian_image)

        # Variance, specifically on edges
        avg = np.mean(laplacian_image)
        rows, cols = laplacian_image.shape
        variance = np.sum((laplacian_image - avg)**2) / (rows * cols)
        print('evaluation value:',variance)

        return variance

    def defocus_hill_search(self, step_size=10, threshold=120, abs_threshold_slope=3):
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
        prev_sharpness, snr, pos, slopes = [], [], [], [] 

        # Set an absolute threshold for adaptive search stride
        abs_threshold_slope = abs_threshold_slope*step_size *1e3  #unit nm
        print('threshold_slope:', abs_threshold_slope)

        # Get the denoising threshold from the initial image
        current_image=self.capture_image_and_save(current_position)
        mean = np.mean(current_image)
        threshold = mean + 2*np.std(current_image)
        print('std:', np.std(current_image))
        print('threshold:', threshold)


        # Search direcrtion determination
        initial_search_start_time = time.time()

        for _ in range(5):
            image = self.capture_image_and_save(current_position)
            denoised_image = cv2.subtract(image, threshold)
            sharpness = self.calculate_sharpness(denoised_image)
            pos.append(current_position)
            prev_sharpness.append(sharpness)

            current_position = current_position + current_step_size
            self.move_stage(current_position)
            prev_sharp = sharpness
            count+=1

        # Linear curve fitting
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

        # Calculate search direction determination time
        initial_search_end_time = time.time()
        search_direction_time= initial_search_end_time - initial_search_start_time
        print(f"Initial search direction determination time: {search_direction_time} seconds")
        
        if direction==-1:
            prev_sharpness, pos, slopes=[],[],[]


        # Start the mountain climbing search, two-step curve fitting
        total_stage_movement_time = 0 
        total_camera_capture_time = 0
        while True:
            count+=1

            stage_movement_start_time = time.time()
            current_position += direction * current_step_size
            self.move_stage(current_position)
            stage_movement_end_time = time.time()
            total_stage_movement_time += (stage_movement_end_time - stage_movement_start_time)

            image_capture_start_time = time.time()
            current_image = self.capture_image_and_save(current_position)
            image_capture_end_time = time.time()
            total_camera_capture_time += image_capture_end_time - image_capture_start_time
            denoised_image = cv2.subtract(current_image, threshold)
            #snr_value = self.calculate_snr(denoised_image)

            # calculate sharpness
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

            # Potential focus position
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
            
            # Adaptive slope threshold
            avg_slope = np.mean(slopes[-3:])
            std_dev_slope = np.std(slopes[-3:])
                        
            print('avg_slope:', avg_slope)
            print('avg_slope - std_dev_slope:', avg_slope + std_dev_slope)
            if avg_slope<0:
                threshold_slope=abs(avg_slope + std_dev_slope)
            else:
                threshold_slope=abs(avg_slope - std_dev_slope)

            # Also set an absolute threshold
            abs_threshold_slope = abs_threshold_slope*step_size*1e3  #unit nm

            # if the slope is flatter, switch to a smaller step size
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

            # if the sharpness value continues to drop, stop the search
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

        print(f"Initial search direction determination time: {search_direction_time} seconds")
        print('total_stage_movement_time1:', total_stage_movement_time)
        print('total_camera_capture_time1:', total_camera_capture_time)

        return current_position, prev_sharpness, pos, peak, total_stage_movement_time, total_camera_capture_time, search_direction_time

    def disconnect(self):
        try:
            #self.core.unload_all_devices()
            self.core.set_focus_device('')
        except Exception as e:
            print('Exception: Closing serial port: ' + str(e))
            

# Define Legendre polynomials
def legendre_polynomials(x, order):
    return np.polynomial.legendre.legval(x, np.eye(order + 1)).T

# Define the error function for least-squares optimization
def error_function(coefficients, x, y, order):
    fitted_curve = np.dot(legendre_polynomials(x, order), coefficients)
    residuals = y - fitted_curve
    return residuals

# For first step curve fitting
def fit_quadratic_curve(x, y, order=2):
    derivative_basis_functions=[]
    # Initial guess for coefficients
    initial_guess =  np.ones(order + 1)
    #print(x)

    # Fit the quadratic curve
    result = least_squares(error_function, initial_guess, args=(x, y, order))

    # Extract fitted coefficients
    fitted_coefficients = result.x

    # Error
    fitted_curve = np.dot(legendre_polynomials(x, order), fitted_coefficients)
    residuals = y - fitted_curve
    mean_squared_error = np.mean(residuals**2)
    print('Error:', mean_squared_error)

    for i in range(1, order + 1):
        derivative_basis_functions.append(i * np.array(x)**(i - 1))
    derivative_basis_functions = np.vstack(derivative_basis_functions).T
    
    # Calculate the derivative of the fitted curve
    derivative_curve = np.dot(derivative_basis_functions, fitted_coefficients[1:])
    print(fitted_coefficients[1:])

    # Can plot the curve if needed
    '''plt.plot(x_shifted, derivative_curve, color='green', label='Derivative Curve')

    plt.xlabel("Defocus distance (um)")
    plt.ylabel("Evaluation value (sharpness)")
    plt.legend()
    plt.show()'''
    
    return fitted_curve, fitted_coefficients, derivative_curve

# Second step curve fitting
def iterative_curve_fitting(x, y, max_order, accuracy_threshold=1e-10):
    order = 1  # Initial order
    coefficients = np.ones(order + 1)
    prev_error = np.inf

    # Max order avoid overfitting
    while order <= max_order:
        result = least_squares(error_function, coefficients, args=(x, y, order))
        fitted_coefficients = result.x
        fitted_curve = np.dot(legendre_polynomials(x, order), fitted_coefficients)

        # Error
        residuals = y - fitted_curve
        mean_squared_error = np.mean(residuals**2)
        print('Error:', abs(mean_squared_error))
        
        # Stop when the error reach the threshold or it starts to increase
        if abs(mean_squared_error) < accuracy_threshold or abs(mean_squared_error) >= abs(prev_error):
            order -=1
            coefficients = np.ones(order+1)
            result = least_squares(error_function, coefficients, args=(x, y, order))
            fitted_coefficients = result.x
            fitted_curve = np.dot(legendre_polynomials(x, order), fitted_coefficients)
            break  
        else:
            prev_error = mean_squared_error
            order += 1  # increase the order for the next iteration
            coefficients = np.ones(order + 1)

    return fitted_curve, fitted_coefficients, order

# Find the curve peak
def find_poly_max(poly):
    neg_poly = lambda x: -poly(x)

    # Use scipy's fmin function to find the maximum
    max_x = fmin(neg_poly, 0)

    return max_x[0]

# Normalise because Legendre polynomials are defined in the range [-1, 1]
def normalise_to_minus_one_to_one(x):
    x_min, x_max = np.min(x), np.max(x)
    x_normalized = 2 * (x - x_min) / (x_max - x_min) - 1
    return x_normalized, x_min, x_max

# Denormalise to get the correct curve
def denormalise_from_minus_one_to_one(x_normalized, x_min, x_max):
    x_original = (x_normalized + 1) / 2 * (x_max - x_min) + x_min
    return x_original

# Second step curve fitting function
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
    plt.savefig('C:/Users/olympus/Desktop/Yuetong/hill1.png')
    # plt.show()

    # Focused position is where the curve reaches its maximum
    poly = Polynomial(best_fit_coefficients)
    focused_position_sharp_normalized = find_poly_max(poly)
    focused_position_sharp = (focused_position_sharp_normalized + 1) / 2 * (x_max - x_min) + x_min
    print(focused_position_sharp)

    return focused_position_sharp


def calculate_niqe(image):
    if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the NIQE score
    niqe_score = niqe(image)

    return niqe_score


if __name__ == '__main__':
    asi_controller = StageController()
    t=0
    if asi_controller.connect_to_stage():
        # This calculate the total autofocus time, which includes four parts.
        start=time.time()
        focused_position, sharpness_values, pos, peak, stage_movement_time, camera_capture_time, search_direction_time =asi_controller.defocus_hill_search()
        print('peak:', peak)
        
        focused_position_sharp= autofocus(focused_position, sharpness_values, pos, cs_threshold=0.9975)

        if abs(focused_position_sharp)>1e4 or focused_position_sharp==0:  # means curve fitting output not reliable
            focused_position_sharp=peak
            t=1  

            print(f"The Sharpness focused position is {focused_position_sharp} um.")
            print("-----cost time:{:.4f}s----".format(time.time()-start))

        # NIQE includes additional camera capture time and stage movement time
        Stage=stage_movement_time
        Capture=camera_capture_time

        if t==0:
            # NIQE comparison to get the final focus prediction
            stage_start_time = time.time()
            asi_controller.move_stage(focused_position_sharp)
            time.sleep(0.1)
            stage_end_time = time.time()
            Stage+= stage_end_time - stage_start_time

            capture_start_time = time.time()
            image1= asi_controller.capture_image_and_save(focused_position_sharp, 100)
            capture_end_time = time.time()
            Capture+= capture_end_time - capture_start_time
            niqe_score1 = calculate_niqe(image1)

            stage_start_time = time.time()
            asi_controller.move_stage(peak)
            time.sleep(0.1)
            stage_end_time = time.time()
            Stage+= stage_end_time - stage_start_time

            capture_start_time = time.time()
            image2= asi_controller.capture_image_and_save(peak, 100)
            capture_end_time = time.time()
            Capture+= capture_end_time - capture_start_time
            niqe_score2 = calculate_niqe(image2)

            print('capture time:', Capture)
            print('stage time:', Stage)

            if niqe_score1>niqe_score2:
                focused_position=peak
                print('The focused image is sharper')
            else:
                focused_position=focused_position_sharp
                print('The focused image is blurrier')
        else:
            focused_position=focused_position_sharp
            time.sleep(0.1)
            image2= asi_controller.capture_image_and_save(focused_position_sharp, 100)

        asi_controller.move_stage(focused_position)
        
        total_autofocus_time=time.time()-start
        #save this information to a file
        with open('Z:/Users/yj368/Autofocus_repeat/time/100ms/11/time.txt', 'w') as f:
            f.write(f'The Sharpness focused position is {focused_position} nm.\n')
            f.write(f"-----cost time:{total_autofocus_time} s----\n")
            f.write(f'search direction determination time: {search_direction_time}\n')
            f.write(f'total camera capture time: {Capture}\n')
            f.write(f'total stage movement time: {Stage}\n')
            f.write(f'autofocus calculation time: {total_autofocus_time-Stage-Capture-search_direction_time}\n')

        asi_controller.disconnect()
    else:
        print('Failed to connect to ASI Stage Controller')



