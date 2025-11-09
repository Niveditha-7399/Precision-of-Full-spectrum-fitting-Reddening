
# Code prepared by Niveditha Parthasarathy, Fall 2024
##################################################################
##################################################################
"""

Takes in the model file and prepares a collection of test spectra 
with the specified amount of reddening and noise. 

"""
# IMPORTING PACKAGES
##################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
#pd.set_option('display.precision', 18)

###################################################################
# SECTION 1: CREATION OF REDDENED SPECTRA
###################################################################
"""

1A: get_A_v(EBV)
1B: wavelength_to_y, calculate_A, new_flux
1C: add_reddening(wavelength,flux,level)
1D: create_reddened_files()
1E: plot_reddened()

"""
###################################################################
########################   1A   ###################################
"""

 Arguments taken: 
 color excess = E(B-V) value [float]

 Process:
 Total extinction in the V-band is calculated by multiplying the 
 extinction parameter R_v and the color excess
 A_v= R_v*E(B-V)

 Returns:
 The total extinction in V-band = A_v [float]


"""
rv=3.2 # value of extinction parameter for LMC
def get_A_v(EBV):
    return rv*EBV

###################################################################
########################   1B   ###################################
"""

Arguments taken:
y= (1/wavelength (micrometre))-1.82 [float]
Index for the E(B-V) list= level [int]

Process:
We use the Cardelli et al. (1989) R_v dependant equation to 
calculate the total extinction at each wavelength, A_lambda

we first calculate the coefficients ax and bx 

ax = 1 + (0.17699*y) 
    - (0.50447*(y**2))
    - (0.02427*(y**3))
    + (0.72085*(y**4))
    + (0.01979*(y**5)) 
    - (0.77530*(y**6)) 
    + (0.32999*(y**7))

bx = (1.41338*y) 
    + (2.28305*(y**2)) 
    + (1.07233*(y**3)) 
    - (5.38434*(y**4)) 
    - (0.62251*(y**5)) 
    + (5.30260*(y**6)) 
    - (2.09002*(y**7))

The argument 'level' is an index, points at an element in the 
list of E(B-V) values. Everytime the function is called, the value 
of level is incremented. 

      level        E(B-V)
        0           0.00
        1           0.01
        2           0.02
        .            .
        .            .
        49          0.5
        
From level, E(B-V) value is obtained. The function then calls the 
get_A_v() function [see 1A] to get A_v value. A_lamba is calculated 
with A_lambda/A_v= ax + (bx/R_v)


Returns: 
Total extinction at y value based on the formula and E(B-V) [float]

"""
#List of E(B-V) values
EBV_list=np.linspace(0,0.5,51)

def wavelength_to_y(wavelength):
    conversion_micro = 1e-4
    exp  = -1
    adds   = -1.82
    b = wavelength * conversion_micro
    y = (b ** exp) + adds
    return y

def calculate_A(wavelength):
    return np.array([wavelength_to_y(w) for w in wavelength])

def new_flux(x0):
    A = calculate_A(x0)
    X = 1 + 0.17699*A - 0.50447*(A**2) - 0.02427*(A**3) + 0.72085*(A**4) + \
        0.01979*(A**5) - 0.77530*(A**6) + 0.32999*(A**7)
    Y = 1.41338*A + 2.28305*(A**2) + 1.07233*(A**3) - 5.38434*(A**4) - \
        0.62251*(A**5) + 5.30260*(A**6) - 2.09002*(A**7)
    Z = X + (Y / 3.2)
    return np.array([X, Y, Z])

###################################################################
########################   1C   ###################################

"""

Arguments taken:
wavelength from the model file [list]
flux from the models file [list]
index of color excess = level [int] [see 1B]

Process:
For each wavelength, x and y are calculated using cardelli et al. (1898) equations 
x = 1/wavelength (micrometres) [conversion => wavelength (angstroms) *1/10000.0]
y = x-1.82
this function calls the A_lambda() function [see 1B] to calculate
reddened flux = original flux x 10^(-A_lambda/2.5)

This is repeated for all wavelengths and the reddened flux is added to a list
which is finally returned

Returns: 
reddened flux values [list]

"""

def add_reddening(x0,y0,level):
    initial_flux = y0
    Z = new_flux(x0)[2]
    EBV_value = EBV_list[level]
    flux = np.array([
        (10**(-0.4*3.2*Z[i]*EBV_value)) * initial_flux[i] for i in range(len(Z))
    ])
    return flux.transpose()

def create_reddened_files(data,str):
    #data=read_model_file(rf'{filename}',6," ")
    column_name=str+" Flux"
    print(f"\ncolumn name= {column_name}")
    x0,y0=data['Wavelength'],data[column_name]
    for level in range(0,51):
        reddened_y=add_reddening(x0,y0,level)
        filename = rf"reddened\{str}_{level}.txt"
        write_lists_to_file(data['Wavelength'], reddened_y, filename)
    print(print(f"\nfilename= {filename}"))
    return x0,y0


###################################################################
########################   1E   ###################################
"""

Arguments taken: 
None

Process:
This function iterates through each reddened file, which are identified
with the 'reddened_file_list' that has stored filenames of reddened files.

It reads the data from each file with read_filesand creates a pandas 
dataframe [see 3A], plotting flux vs wavelength. A variable 'EBV_count' is 
used to access elements from 'EBV_list' just like 'level' [see 1B]

Dataframes are deleted after each iteration to ensure there are no memory leaks.

Returns:
None

"""

def plot_reddened():
    EBV_count=0
    for file in reddened_file_list:
        if EBV_count==50:
            EBV_count=0
        data=read_file(rf'{file}',2,'\t')
        plt.plot(data['Wavelength'],  data['Flux'],label=f"E(B-V)={EBV_list[EBV_count]}")
        EBV_count+=1
        del data
    return None

###################################################################
# SECTION 2: CREATION OF NOISE IN REDDENED SPECTRA
###################################################################
"""

2A: add_noise(flux, noise_mean)
2B: create_noise_reddened_files()

"""
###################################################################
########################   2A   ###################################
"""

Arguments taken: 
flux of reddened spectra [list]
Mean for Gaussian distribution = noise_mean

Process:

this function uses Gaussian distribution to describe noise. It can be 
adjusted through the mean and standard deviation. The former changes 
the amount of noise and the latter affects the intensity of noise. 

noise_mean holds the mean value to be used in the gaussian 
distribution creation. The value for it comes from the list 'noise_means_list' 
when create_noise_reddened_files() is called

Returns:
list of flux values with noise

"""

def add_noise(flux, noise_mean):
    noise = np.random.normal(0, noise_mean, size=len(flux))
    return np.array(flux) + noise

###################################################################
########################   2B   ###################################
"""

Arguments taken: 
None

Process:
It is used once in the main script. It iterates through each reddened 
file using the stored filenames 'reddened_file_list' [see 1E], calls read_file
[see 3A]. 

For each such file it iterates over 10 different values of 'noise_level'
and calls add_noise [see 2A]. This data is then written to files with 
write_lists_to_file() [see 3B]. Variable 'level_file' works the same 
way as 'level' referring to index of E(B-V) values but here it is
used for filename creation. The dataframe crated when reading 
is deleted after each iteration to ensure there are no memory leaks.

Returns:
None

"""

def create_noise_reddened_files(str,num_duplicates):
    reddening_level=0 # initialize reddening level

    # Each file in reddened files are opened
    for file in reddened_file_list: 
        data=read_file(rf'reddened\{file}',2,'\t')

        # For each reddened spectra, we loop through 10 noise levels
        for SNR in range(10,101,10):

            # the number of reps (same SNR and reddening)
            for reps in range(1,num_duplicates+1):

                    # Generate random noise with the calculated standard deviation
                    noise = np.random.normal(loc=0, scale=data['Flux'] / SNR)
                    noisy_flux=data['Flux']+noise
                    filename = rf"noise_reddened\{str}_{reddening_level}_{int(SNR/10)}_{reps}.txt"
                    write_lists_to_file(data['Wavelength'], noisy_flux, filename)

        del data
        reddening_level+=1 # goes upto 50
        if reddening_level>50:
            break

    return None
###################################################################
# SECTION 3: READING AND PLOTTING
###################################################################
"""

3A: read_file(filename,skiplines,seperator)
3B: write_lists_to_file(x_list, y_list, filename)
3C: setting_plot_paramters_and_display()
3D: read_model_file(filename,skiplines,seperator)
"""
###################################################################
########################   3A   ###################################
"""

Arguments taken: 
file location/file name= filename [string]
number of lines to skip= skiplines [int]
seperator between data in the file read= seperator [string]

Process:
Reads the datafile using teh file location, skipping the lines specified
and uses the separator to differentiate the flux values and allocates them
into columns.

It is called within the functions: 
create_reddened_files() and plot_reddened()

Returns:
a pandas dataframe with the wavelength and flux values [dataframe]


"""
columns=['Wavelength','Flux']
def read_file(filename,skiplines,seperator):
    data = pd.read_csv(rf"{filename}",skiprows=skiplines,sep=seperator, names=columns,engine='python', dtype={'Wavelength': 'float64', 'Flux': 'float64'})
    return data

###################################################################
########################   3B   ###################################

"""

Arguments taken: 
Wavelength values = x_list [list]
Flux values = y_values [list]
file location/ file name= filename [string]

Process:
Writes two lists (wavelength and fluxes) to a specified file 
which will be created if it doesnt exists; will be overwritten if it exists.

X_Num1 and X_NUM2 are values indicating minimum and maximum values of wavelength range
Only the wavelength range defined by X_NUM1 and X_NUM2 will be written


It is called within the functions: 
create_reddened_files() and create_noise_reddened_files()

Returns:
None

"""

X_NUM1=3700 # starting wavelength value
X_NUM2=5001 # ending wavelength value

def write_lists_to_file(x_list, y_list, filename): # Writes two lists to a file in two columns.
              try:
                    with open(filename, 'w') as file:
                        file.write(f"# Output file name = {filename}\n")
                        file.write("# Lambda(A)       Flux   \n")
                        for x, y in zip(x_list, y_list):
                            if x>=X_NUM1 and x<X_NUM2:
                                file.write(f"{x}\t{y}\n")
                    print(f"File '{filename}' created successfully.")
              except Exception as e:
                    print(f"Error creating file:{e}")

              return None


###################################################################
########################   3C   ###################################
"""

Arguments taken: 
None

Process:
This function sets plot parameters like x-axis and y-axis limits, labels and title

Returns:
None


"""
def setting_plot_paramters_and_display():
    plt.xlim(X_NUM1,X_NUM2)
    plt.ylim(0.0,0.011)
    #plt.legend()
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux')
    plt.title('Spectral Plot')
    plt.show()
    return None

###################################################################
########################   3D   ###################################
"""

Arguments taken: 
filepath/file name= filename [string]
number of lines to skip= skiplines [int]
seperator between data in the file read= seperator [string]

Process:
Reads the datafile using the file location, skipping the lines specified
and uses the separator to differentiate the flux values corresponding to 
different ages and allocates them into columns.

Returns:
a pandas dataframe with the wavelength and flux values for all ages[dataframe],
and ages [list]


"""

def read_model_file(filename,skiplines,seperator):
    with open(filename, 'r') as f:
        header_line = f.readline().split(',')
        f.close()
    header_line[-1] = header_line[-1].strip()
    print(header_line)
    columns=['Wavelength']
    for i in header_line:
        columns.append(i+" Flux")
    print(columns)
    data = pd.read_csv(rf"{filename}",skiprows=skiplines,sep=seperator, names=columns,engine='python')
    return data, header_line


###################################################################
###################################################################
# SECTION 4: MAIN SCRIPT
###################################################################
input_file_list = glob.glob("InputData/*.spec")
import os
input_model_list= glob.glob("InputModel/*.spec")
df,ages= read_model_file(r"C:\Users\nived\OneDrive\Desktop\spicy depression\computational physics\NoiseEasy\NoiseEasy\InputModel\compiled_model",1," ")
print(df)
print(EBV_list)

for str in ages:
    x0,y0=create_reddened_files(df,str)
#    plot_reddened()
#    plt.plot(x0, y0,color='purple', label="real")
#    setting_plot_paramters_and_display()
#reddened_file_list = glob.glob("reddened/*.txt")

files = os.listdir("reddened")
new_ages=['# 6.0', ' 6.5', ' 6.7', ' 6.8', ' 6.9', ' 7.0', ' 7.16', ' 7.4', ' 7.6', ' 7.7', ' 8.0', ' 8.2', ' 8.5', ' 8.7', ' 8.9', ' 9.1', ' 9.16', ' 9.4', ' 9.6', ' 9.8', ' 9.9', ' 10.0', '10.1']
for str in new_ages:
    reddened_file_list = [file for file in files if file.startswith(str) and file.endswith(".txt")]
    create_noise_reddened_files(str,20)


