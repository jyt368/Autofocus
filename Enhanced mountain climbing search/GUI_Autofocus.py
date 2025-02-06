from tkinter import *
from tkinter import messagebox
import time
from pycromanager import Core
from Autofocus_widefield import StageController,autofocus, calculate_niqe
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from matplotlib.colors import LinearSegmentedColormap

# if "ok" button is clicked
def on_button_click():
    step_get=int(step.get())
    slope_thr= int(thres.get())
    asi_controller = StageController(path='.../our method/test')
    if asi_controller.connect_to_stage():
        # test move_stage--correct
        # asi_controller.move_stage(1000)  #unit nm
        # asi_controller.get_stage_position()
        start=time.time()
        focused_position, sharpness_values, pos, peak =asi_controller.defocus_hill_search(step_size=step_get, abs_threshold_slope=slope_thr)
        print('peak:', peak)
        
        focused_position_sharp= autofocus(focused_position, sharpness_values, pos)

        if focused_position_sharp>1e5:
            focused_position_sharp=peak

        print(f"The Sharpness focused position is {focused_position_sharp} um.")
        print("-----cost time:{:.4f}s----".format(time.time()-start))

        asi_controller.move_stage(int(focused_position_sharp))
        image1= asi_controller.capture_image_and_save(focused_position_sharp)
        niqe_score1 = calculate_niqe(image1)
        print(niqe_score1)
        asi_controller.move_stage(int(peak))
        image2= asi_controller.capture_image_and_save(peak)
        niqe_score2 = calculate_niqe(image2)
        print(niqe_score2)
        if niqe_score1>niqe_score2:
            focused_position=peak
            print('The focused image is sharper')
        else:
            focused_position=focused_position_sharp
            print('The focused image is blurrier')

        asi_controller.move_stage(int(focused_position))

        messagebox.showinfo(title='Successfully autofocused!', 
                            message= f"The Sharpness focused position is {focused_position} µm.\n----- Cost time: {time.time() - start:.4f}s -----")

        # Display or not
        if display_default.get() == "Final Image Shown":
            core = Core()
            core.snap_image()
            result = core.get_tagged_image()
            pixels = np.squeeze(np.reshape(result.pix,newshape=[-1, result.tags["Height"], result.tags["Width"]],))
            #plt.imshow(pixels, cmap='gray')
            # If you want to display image in different color, here's an example
            cmap_green = LinearSegmentedColormap.from_list('black_to_green', [(0, 0, 0), (0, 1, 0)])
            plt.imshow(pixels, cmap=cmap_green)
            plt.axis('off')
            plt.rcParams['toolbar']='None'
            
            plt.show()

        asi_controller.disconnect()
    else:
        print('Failed to connect to ASI Stage Controller')

def cancel_button_click():
    window.destroy()  

'''--main code--'''
# GUI interface title
window = Tk()
window.title("Autofocus Parameter Configuration")
# (width x height)
window.geometry('512x256')
window.tk.call('tk', 'scaling', 3.0)

# Other parameters are defined via micromanager (exposure, channel, objective, etc.) 
# Parameter1: Step Size
label_1 = Label(window, text="Step Size:")
label_1.grid(column=0, row=0)
# Default value is 1 um
step_default = StringVar(value='1')
step = Spinbox(window, from_=1, to=100, textvariable=step_default, width=10)
step.grid(column=1, row=0)

# Parameter2: Slope threshold 
label_2 = Label(window, text="Threshold:")
label_2.grid(column=0, row=1)
# Default value is 3, empirical
thres_default = StringVar(value='3')
thres = Spinbox(window, from_=1, to=10, textvariable=thres_default)
thres.grid(column=1, row=1)

# Choose to display the final focused image or not
label_3 = Label(window, text="Display:")
label_3.grid(column=0, row=2)
# Default
display_default=StringVar(value="Final Image Shown")
display_options = ["Final Image Shown", "Final Image Not Shown"]
display_opt = OptionMenu(window, display_default, *display_options)
display_opt.grid(column=1, row=2)

# Add a label to display messages--> autofocus time and position output
label = Label(window, text="", font=("Arial", 10))

# Add buttons
bt_ok = Button(window, text="Ok", command=on_button_click)
bt_ok.grid(column=0, row=4)
bt_cancel = Button(window, text="Cancel", command=cancel_button_click)
bt_cancel.grid(column=1, row=4)

# Run the GUI event loop
window.mainloop()
