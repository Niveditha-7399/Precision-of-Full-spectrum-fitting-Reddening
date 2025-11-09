#########################################################
"""

This code processes spectral data from multiple model files, 
compiles it into a single dataframe, and then exports the compiled data 
to a text file compatible with the Analyzer of Spectra for Age Determination.

It takes 23 model spectra files corresponding to different ages within 
6.0 < log (age/yr) < 10.1. The 
wavelength range of the compiled model is adjusted from the initial range 
91 Å -1,600,000 Å to 3700 Å -5000 Å.

"""
#########################################################
# IMPORTING PACKAGES

import pandas as pd 
pd.set_option('display.precision', 16) 
columns=['Wavelength','Flux'] 


######################################################### 
######################################################### 
""" 
This function reads spectral data from a model file, skips a  
specified number of lines and returns a dataframe with the spectral data 
""" 
######################################################### 

def read_file(filename,skiplines,seperator,nrows): 
    data = pd.read_csv(rf"{filename}",skiprows=skiplines,sep=seperator, 
    names=columns,engine='python',nrows=nrows,dtype={'Flux': 'float64'}) 
    return data 


# list of the identifications of the model files 
file_n=['020','045','055','061','067','070','078','090', 
'104','110','116','120','125','130','135','138', 
'139','150','158','166','171','181','193'] 
df = pd.DataFrame(columns=file_n) 

######################################################### 
######################################################### 

""" 
For each model file, create a column of its fluxes in the 
dataframe called df. It calls the read_file() function to  
get the fluxes and wavelength as a dataframe 
""" 

######################################################### 

for Num in file_n: 
f=rf"models\bc2003_hr_m22_chab_ssp_{Num}.spec" 
data=read_file(f,731,'  ',1301) 
df[Num]=data['Flux'] 
print(f"\n\n\n\n\n\ndata:{data}\n\n\nflux:{data['Flux'][0]}") 
if Num=='193': 
    df.index=data['Wavelength'] 
del data 

######################################################### 
######################################################### 

""" 
From the dataframe created, create a model file called  
compiled_model compatible with Analyzer of Spectra for Age Determination
""" 
######################################################### 

def create_model(filename,df): 
    header='# 6.0, 6.5, 6.7, 6.8, 6.9, 7.0, 7.16, 7.4, 7.6, 7.7, 8.0, 8.2, 8.5, 8.7, 8.9, 9.1, 9.16, 9.4, 9.6, 9.8, 9.9, 10.0, 10.1\n' 
    with open(filename, 'w') as f: 
    # Write the first line 
    f.write(header) 
    for index, row in df.iterrows(): 
        # Create a string with the index and column values separated by spaces 
        line = f"{index} {' '.join(map(str, row.values))}\n" 
        f.write(line) 
    f.close() 

create_model("compiled_model.txt",df)