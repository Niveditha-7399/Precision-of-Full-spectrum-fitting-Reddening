
# Code written by Niveditha Parthasarathy, Fall 2024
################################################################
################################################################
"""

This is the analysis part of the project. We will be reading 
results from Analyzer of Spectra for Age Determination and store 
them in a xlsx file

This code will require the results files to be stored in the same
directory as the code's source file

"""
################################################################

# IMPORTING PACKAGES
import openpyxl
import pandas as pd
import numpy as np

################################################################
############################################################
# Reads results from multiple text files (results_1.txt to results_20.txt by default).
# looping through all the results files
for i in range(1,20):
    file_path = rf"results_{i}.txt"
    nrows=11730

    # create a datafrme containing the results data
    def read_rows(file_path):
        data = []
        with open(file_path, 'r') as f:
            # Skip the header line
            next(f)
            # Extracts relevant information: Test_Age, Reddening, SNR, Retrieved_age, Retrieved_reddening.
            for _ in range(nrows):
                line = f.readline().strip().split()
                print(line)
                if line[0]=="#":
                    line=line[1:]
                first_part, second_part,third_part,trashed = line[0].split('_', 3) 
                line[0] = first_part 
                line.insert(1, second_part)  
                line.insert(2, third_part)  
                data.append(line)

        # Creates a pandas DataFrame to store the extracted data.
        df = pd.DataFrame(data)
        return df

    # the results dataframe
    df = read_rows(file_path)
    df=df.iloc[:, [0,1,2,3,4]]
    # Cleans and prepares the data:
    # Assigns column names.
    df.columns = ['Test_Age', 'Reddening','SNR','Retrieved_age','Retrieved_reddening'] # These will be the column names
 
    # Converts data types to floats.
    df['Test_Age'] = df['Test_Age'].astype(float)
    df['Reddening'] = df['Reddening'].astype(float)
    df['SNR'] = df['SNR'].astype(float)
    df['Retrieved_age'] = df['Retrieved_age'].astype(float)
    df['Retrieved_reddening'] = df['Retrieved_reddening'].astype(float)
    # Adjusts Reddening and SNR values as needed.
    df['Reddening'] = (df['Reddening'])*0.5/50
    df['SNR'] = df['SNR']*10.0

    df_sorted = df.sort_values(by='Reddening')
    df_sortedn=df_sorted.reset_index(drop=True)

    # Appends the sorted data to an existing Excel file 
    # ("Compiled_results.xlsx") or creates a new file if it doesn't exist.
    file_name = 'Compiled_results.xlsx'
    try:
        # Here, previous entries will not overwritten
        existing_df = pd.read_excel(file_name)
        df_sortedn = pd.concat([existing_df, df_sortedn], ignore_index=True)

    except FileNotFoundError:
        # In the first run, the file will not exist
        pass

    df_sortedn.to_excel(file_name, index=False)






