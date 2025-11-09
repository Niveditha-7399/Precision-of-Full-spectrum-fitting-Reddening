##########################################################################
##########################################################################

"""
This code reads the excel file containing all the results data and creates
different plots for analyzing the retrieved reddening. The code for plots 
used in the thesis are provided first, followed by code for creating an
excel sheet with the percentage error of retrieved reddening across all
SNRs, ages and reddening. 

Then code for some supplementary plots is provided

This code requires the excel sheet to be saved in the same directory as its
source code
"""

###########################################################################
###########################################################################


###########################################################################
# IMPORTING PACKAGES
###########################################################################
###########################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import seaborn as sns
from matplotlib.markers import MarkerStyle


# read the file containing the Analyzer of Spectra for Age Determination results and store it in df
file_path = "Compiled_results.xlsx"
df = pd.read_excel(file_path, header=0, usecols="A:E", nrows=234600)
rcParams['font.family'] = 'Times New Roman' 
rcParams['font.size'] = 12  


#########################################################################
#########################################################################
# Figure 1: Standard Deviation of Retrieved Reddening across SNRs and Test Reddening
#########################################################################
def std_snr_red(df):
    # Group the data by SNR and Reddening, calculating the standard deviation
    std_dev_by_snr_and_reddening = df.groupby(['SNR', 'Reddening'])['Retrieved_reddening'].std().reset_index()

    pivot_table = std_dev_by_snr_and_reddening.pivot(index='SNR', columns='Reddening', values='Retrieved_reddening')

    # Choose a colormap
    cmap = 'viridis'  

    ax = sns.heatmap(pivot_table, cmap=cmap, cbar_kws={'label': '\nStandard Deviation'},linewidths=0.2)

    # Customize the plot
    ax.set_yticklabels(pivot_table.index, rotation=0)
    ax.set_xlabel('Test Reddening', labelpad=10)  
    ax.set_ylabel('Signal-to-Noise Ratio', labelpad=10) 
    ax.invert_yaxis()

    plt.title('Figure 1: Standard Deviation of Retrieved Reddening \nAcross Signal-to-Noise Ratios and Test Reddening',pad=20,linespacing=1.7)
    plt.tight_layout()
    plt.show()
    return None


#############################################################################
#############################################################################
# FIGURE 2: Stanadrd deviation of retrieved reddening vs reddening for different ages
#############################################################################

def std_subparts(age_lists,snr_list,df):
   
    count = 0
    # Create a colormap

    marker_map = {
        100: 's',
        90: 'h',
        80: 'p',
        70: 'd',
        60: '^',
        50: 'o',
        40: 'x',
        30: '*',
        20: '+',
        10: MarkerStyle('.')  # Custom dot marker
    }

    for i in age_lists:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
        axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easier iteration
        count = 0
        for age in i:
            df_a=df[df['Test_Age'] == age]
            # Create the scatter plot with markers based on SNR
            for snr_value, marker_style in marker_map.items():
                subset = df_a[df_a['SNR'] == snr_value]
                axes[count].scatter(subset['Reddening'], subset['Retrieved_reddening'], alpha=0.7, marker=marker_style, label=f'SNR={snr_value}')
            axes[count].set_title(f'Figure 2: Standard Deviation of \nretrieved Reddening vs. Test Reddening, Age: {age}')
            axes[count].set_xlabel('Test Reddening')
            axes[count].set_ylabel('Standard Deviation')
            axes[count].grid(True)
            axes[count].set_ylim(0.0,0.25)
            #axes[count].legend()
            count += 1

        plt.tight_layout()
        plt.show()
    return None



#############################################################################
#############################################################################
# FIGURE 3 Stanadrd deviation of retrieved reddening vs reddening for different ages 
# for SNR>20
#############################################################################

def std_subparts_above20(age_lists,snr_list,df):
    count = 0

    marker_map = {
        100: 's',
        90: 'h',
        80: 'p',
        70: 'd',
        60: '^',
        50: 'o',
        40: 'x',
        30: '*'
    }

    for i in age_lists:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
        axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easier iteration
        count = 0
        for age in i:
            df_a=df[df['Test_Age'] == age]
            # Create the scatter plot with markers based on SNR
            for snr_value, marker_style in marker_map.items():
                subset = df_a[df_a['SNR'] == snr_value]
                axes[count].scatter(subset['Reddening'], subset['Retrieved_reddening'], alpha=0.7, marker=marker_style, label=f'SNR={snr_value}')
            axes[count].set_title(f'Figure 3 Standard Deviation of \nretrieved Reddening vs. Test Reddening, with SNR > 20, for Age: {age}')
            axes[count].set_xlabel('Test Reddening')
            axes[count].set_ylabel('Standard Deviation')
            axes[count].grid(True)
            axes[count].set_ylim(0.0,0.15)
            #axes[count].legend()
            count += 1

        plt.tight_layout()
        plt.show()
    return None


#############################################################################
#############################################################################
# APPENSIX A figures: Error in retreived reddening vs reddening for different ages
#############################################################################

def err_parts(age_lists,snr_list,df):
   
    count = 0
    # Create a colormap
    colors = plt.cm.tab20(np.linspace(0, 1, len(snr_list)))

    for i in age_lists:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
        axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easier iteration
        count = 0
        for age in i:
            for idx, snr in enumerate(snr_list):
                df_mean_err = df.groupby(['Test_Age', 'Reddening', 'SNR'], as_index=False)['Reddening_Prediction_Error'].mean()
                df_max = df.groupby(['Test_Age', 'Reddening', 'SNR'], as_index=False)['Reddening_Prediction_Error'].max()
                df_min = df.groupby(['Test_Age', 'Reddening', 'SNR'], as_index=False)['Reddening_Prediction_Error'].min()
                max_df = df_max[df_max['Test_Age'] == age]
                min_df = df_min[df_min['Test_Age'] == age]

                max_df = max_df[max_df['SNR'] == snr]
                min_df = min_df[min_df['SNR'] == snr]

                df_mean_err = df_mean_err[df_mean_err['Test_Age'] == age]
                df_mean_err = df_mean_err[df_mean_err['SNR'] == snr]

                max_df['max_err'] = max_df['Reddening_Prediction_Error'] - df_mean_err['Reddening_Prediction_Error'] 
                min_df['min_err'] = df_mean_err['Reddening_Prediction_Error'] - min_df['Reddening_Prediction_Error']

                yerr_max = max_df['max_err'].values
                yerr_min = min_df['min_err'].values

                print("\n\n", yerr_min, yerr_max)

                axes[count].errorbar(df_mean_err['Reddening'], df_mean_err['Reddening_Prediction_Error'], 
                                     yerr=[yerr_min, yerr_max], fmt='o', color=colors[idx], label=f'SNR= {snr}',alpha=0.7)

            axes[count].set_title(f'Error in retrieved reddened versus \n reddening for Age: {age}')
            axes[count].set_xlabel('Test Reddening')
            axes[count].set_ylabel('Retreived Reddening- Test Reddening')
            axes[count].grid(True)
            axes[count].legend()
            count += 1

        plt.tight_layout()
        plt.show()
    return None


##########################################################################
##########################################################################
# working space
##########################################################################

grouped_df_mean = df.groupby(['Test_Age', 'Reddening', 'SNR'])['Retrieved_reddening'].mean().reset_index()
grouped_df_above50_mean = grouped_df_mean[grouped_df_mean['SNR'] >= 50]
grouped_df_below50_mean = grouped_df_mean[grouped_df_mean['SNR'] < 50]
grouped_df_mean['Reddening_Prediction_Error'] = grouped_df_mean['Retrieved_reddening'] - grouped_df_mean['Reddening']

grouped_df_std = df.groupby(['Test_Age', 'Reddening', 'SNR'])['Retrieved_reddening'].std().reset_index()
grouped_df_above50_std = grouped_df_std[grouped_df_std['SNR'] >= 50]
grouped_df_below50_std = grouped_df_std[grouped_df_std['SNR'] < 50]


# $$$$$$$$ #

df['Reddening_Prediction_Error'] = df['Retrieved_reddening'] - df['Reddening']

df_mean_err=df.groupby(['Test_Age', 'Reddening', 'SNR'], as_index=False)['Reddening_Prediction_Error'].mean()
df_max=df.groupby(['Test_Age', 'Reddening', 'SNR'], as_index=False)['Reddening_Prediction_Error'].max()
df_min=df.groupby(['Test_Age', 'Reddening', 'SNR'], as_index=False)['Reddening_Prediction_Error'].min()

age_list=df_mean_err['Test_Age'].unique()
snr_list=df_mean_err['SNR'].unique()
age_lists = [age_list[0:4], age_list[4:8], age_list[8:12], age_list[12:16], age_list[16:20], age_list[20:24]]



std_snr_red(df)
err_parts(age_lists,snr_list,df)
std_subparts(age_lists,snr_list,grouped_df_std)
std_subparts_above20(age_lists,snr_list,grouped_df_std)


#######################################################################
#######################################################################
# Create a table with the percentage error in reddening retrievals
# for each SNR, age and reddening combination
#######################################################################

def calculate_percentage_error(df, group_by_vars):
    results = []
    grouped = df.groupby(group_by_vars)
    
    for group, data in grouped:
        real_reddening_mean = data['Reddening'].mean()
        if real_reddening_mean != 0:  # Avoid division by zero
            mean_error = (data['Reddening_Prediction_Error'].mean() / real_reddening_mean) * 100  # Convert to percentage
            result = {var: val for var, val in zip(group_by_vars, group)}
            result.update({'Percentage Error': f"{mean_error:.2f}%"})
            results.append(result)
        else:  # Handle division by zero case
            result = {var: val for var, val in zip(group_by_vars, group)}
            result.update({'  '})
            results.append(result)
                
    return pd.DataFrame(results)

group_by_vars = ['Test_Age', 'Reddening', 'SNR']
percentage_error_df = calculate_percentage_error(df, group_by_vars)
percentage_error_df.to_excel('comprehensive_percentage_error.xlsx', index=False)
print("Comprehensive DataFrame successfully written to comprehensive_percentage_error.xlsx")


##########################################################################
##########################################################################
# SUPPLEMENTARY PLOTS (Not included in thesis):


##########################################################################
##########################################################################
# plot of the mean reddening retrieved vs reddening
##########################################################################
def mean_all(grouped_df):

    marker_map = {
        100: 's',
        90: 'h',
        80: 'p',
        70: 'd',
        60: '^',
        50: 'o',
        40: '*',
        30: 'x',
        20: '+',
        10: '|'
    }
    plt.plot(grouped_df['Reddening'],grouped_df['Reddening'],label='True Reddening = \nRetrieved Reddening',c='k',linewidth=0.3)
    for snr_value, marker_style in marker_map.items():
        subset = grouped_df[grouped_df['SNR'] == snr_value]
        plt.scatter(subset['Reddening'], subset['Retrieved_reddening'], c=subset['Test_Age'], cmap='viridis', alpha=0.7, marker=marker_style, label=f'SNR={snr_value}')

    plt.xlabel('Test Reddening')
    plt.ylabel('Average Retrieved Reddening')
    plt.title('Scatter Plot of Average Retrieved Reddening vs. Test Reddening, Colored by Ages, with SNR Markers')
    plt.legend()
    plt.colorbar(label='Ages')
    plt.tight_layout()
    plt.show()
    return None


##########################################################################
##########################################################################
# plot of the mean reddening retrieved vs reddening but in 3 SNR groupings
# 1. SNR= 10,100 2. SNR>40 3. SNR<50
##########################################################################
def mean_parts(grouped_df_above50,grouped_df_below50):
    marker_map = {
        100: 's',
        10: '*'
    }

    plt.plot(grouped_df_above50['Reddening'],grouped_df_above50['Reddening'],label='True Reddening = \nRetrieved Reddening',c='k',linewidth=0.3)
    for snr_value, marker_style in marker_map.items():
        subset = grouped_df_above50[grouped_df_above50['SNR'] == snr_value]
        linewidth = 0.3
        plt.scatter(subset['Reddening'], subset['Retrieved_reddening'], c=subset['Test_Age'], cmap='viridis', alpha=0.7, marker=marker_style, label=f'SNR={snr_value}',edgecolors='black', linewidth=linewidth)

    plt.xlabel('Reddening')
    plt.ylabel('Average Retrieved Reddening')
    plt.title('Scatter Plot of Average Retrieved Reddening vs. Test Reddening, Colored by Ages, with SNR Markers')
    plt.colorbar(label='Ages')
    plt.legend()
    plt.show()

    marker_map = {
        100: 's',
        90: 'h',
        80: 'p',
        70: 'd',
        60: '^',
        50: 'd'
    }


    plt.plot(grouped_df_above50['Reddening'],grouped_df_above50['Reddening'],label='True Reddening = \nRetrieved Reddening',c='k',linewidth=0.3)
    for snr_value, marker_style in marker_map.items():
        subset = grouped_df_above50[grouped_df_above50['SNR'] == snr_value]
        if snr_value >= 50|snr_value == 30|snr_value == 0:
            linewidth = 0.3
        else:
            linewidth = 0.8
        plt.scatter(subset['Reddening'], subset['Retrieved_reddening'], c=subset['Test_Age'], cmap='viridis', alpha=0.7, marker=marker_style, label=f'SNR={snr_value}',edgecolors='black', linewidth=linewidth)

    plt.xlabel('Reddening')
    plt.ylabel('Average Retrieved Reddening')
    plt.title('Scatter Plot of Average Retrieved Reddening vs. Test Reddening, Colored by Ages, with SNR Markers')
    plt.colorbar(label='Ages')
    plt.legend()
    plt.show()

    marker_map =     {
        40: '*',
        30: 'x',
        20: '+',
        10: '|',
    }

    plt.plot(grouped_df_below50['Reddening'],grouped_df_below50['Reddening'],label='True Reddening = \nRetrieved Reddening',c='k',linewidth=0.3)
    # Create the scatter plot with markers based on SNR
    for snr_value, marker_style in marker_map.items():
        subset = grouped_df_below50[grouped_df_below50['SNR'] == snr_value]
        if snr_value >= 50|snr_value == 30:
            linewidth = 0.3
        else:
            linewidth = 0.6
        plt.scatter(subset['Reddening'], subset['Retrieved_reddening'], c=subset['Test_Age'], cmap='viridis', alpha=0.7, marker=marker_style, label=f'SNR={snr_value}',edgecolors='black', linewidth=linewidth)

    plt.xlabel('Reddening')
    plt.ylabel('Average Retrieved Reddening')
    plt.title('Scatter Plot of Average Retrieved Reddening vs. Test Reddening, Colored by Ages, with SNR Markers')
    plt.legend()
    plt.colorbar(label='Ages')
    plt.show()
    return None


###########################################################################
###########################################################################
# plot of standard deviation of retrieved reddening vs reddening
# groupings: for all SNRS and SNR=10,100
###########################################################################

def std_all(grouped_df):

    marker_map = {
        100: 's',
        90: 'h',
        80: 'p',
        70: 'd',
        60: '^',
        50: 'o',
        40: 'x',
        30: '*',
        20: '+',
        10: MarkerStyle('.')  # Custom dot marker
    }

    # Create the scatter plot with markers based on SNR
    for snr_value, marker_style in marker_map.items():
        subset = grouped_df[grouped_df['SNR'] == snr_value]
        plt.scatter(subset['Reddening'], subset['Retrieved_reddening'], c=subset['Test_Age'], cmap='viridis', alpha=0.7, marker=marker_style, label=f'SNR={snr_value}')

    # Customize the plot
    plt.xlabel('Test Reddening')
    plt.ylabel('Standard Deviation')
    plt.title('Scatter Plot of Standard deviation of Retrieved Reddening vs. Test Reddening, Colored by Ages, with SNR Markers')
    plt.legend()
    plt.colorbar(label='Ages')
    plt.show()

    marker_map = {
        100: 's',
        10: '*'
    }

    for snr_value, marker_style in marker_map.items():
        subset = grouped_df[grouped_df['SNR'] == snr_value]

        linewidth = 0.3

        plt.scatter(subset['Reddening'], subset['Retrieved_reddening'], c=subset['Test_Age'], cmap='viridis', alpha=0.7, marker=marker_style, label=f'SNR={snr_value}',edgecolors='black', linewidth=linewidth)

    plt.xlabel('Test Reddening')
    plt.ylabel('Standard Deviation')
    plt.title('Scatter Plot of Standard deviation of Retrieved Reddening vs. Test Reddening, Colored by Ages, with SNR Markers')
    plt.legend()
    plt.colorbar(label='Ages')
    plt.show()
    return None

###########################################################################
###########################################################################
# plot of standard deviation of retrieved reddening vs reddening
# groupings: for SNR=10,100, SNR>40, SNR<50
###########################################################################

def std_parts(grouped_df_above50,grouped_df_below50):
    marker_map = {
        100: 's',
        10: '*'
    }

    for snr_value, marker_style in marker_map.items():
        subset = grouped_df_above50[grouped_df_above50['SNR'] == snr_value]

        linewidth = 0.3

        plt.scatter(subset['Reddening'], subset['Retrieved_reddening'], c=subset['Test_Age'], cmap='viridis', alpha=0.7, marker=marker_style, label=f'SNR={snr_value}',edgecolors='black', linewidth=linewidth)

    plt.xlabel('Test Reddening')
    plt.ylabel('Standard Deviation')
    plt.title('Scatter Plot of Standard deviation of Retrieved Reddening vs. Test Reddening, Colored by Ages, with SNR Markers')
    plt.legend()
    plt.colorbar(label='Ages')
    plt.show()
    marker_map = {
        100: 's',
        90: 'h',
        80: 'p',
        70: '*',
        60: 'x',
        50: '+'
    }

    for snr_value, marker_style in marker_map.items():
        subset = grouped_df_above50[grouped_df_above50['SNR'] == snr_value]
        if snr_value >= 50:
            linewidth = 0.2
        else:
            linewidth = 0.8
        plt.scatter(subset['Reddening'], subset['Retrieved_reddening'], c=subset['Test_Age'], cmap='viridis', alpha=0.7, marker=marker_style, label=f'SNR={snr_value}',edgecolors='black', linewidth=linewidth)

    plt.xlabel('Test Reddening')
    plt.ylabel('Standard Deviation')
    plt.title('Scatter Plot of Standard deviation of Retrieved Reddening vs. Test Reddening, Colored by Ages, with SNR Markers')
    plt.legend()
    plt.colorbar(label='Ages')
    plt.show()

    marker_map =     {
        40: '*',
        30: 'x',
        20: '+',
        10: '|'
    }

    for snr_value, marker_style in marker_map.items():
        subset = grouped_df_below50[grouped_df_below50['SNR'] == snr_value]
        if snr_value >= 50|snr_value == 30:
            linewidth = 0.3
        else:
            linewidth = 0.4
        plt.scatter(subset['Reddening'], subset['Retrieved_reddening'], c=subset['Test_Age'], cmap='viridis', alpha=0.7, marker=marker_style, label=f'SNR={snr_value}',edgecolors='black', linewidth=linewidth)

    plt.xlabel('Test Reddening')
    plt.ylabel('Standard Deviation')
    plt.title('Scatter Plot of Standard deviation of Retrieved Reddening vs. Test Reddening, Colored by Ages, with SNR Markers')
    plt.legend()
    plt.colorbar(label='Ages')
    plt.show()
    return None

#mean_all(grouped_df_mean)
#mean_parts(grouped_df_above50_mean,grouped_df_below50_mean)
#std_all(grouped_df_std)
#std_parts(grouped_df_above50_std,grouped_df_below50_std)


