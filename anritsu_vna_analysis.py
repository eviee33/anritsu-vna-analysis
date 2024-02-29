import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import re

#creates a pandas table of the csv data and averages every x data points
def retrieve_data(file_path, group_size):
    starting_row = None
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        #checks which format the csv file is in
        if 'Title' in first_line:
            for i, line in enumerate(file):
                #finds where the data begins
                if 'Frequency(MHz)' in line.split(',')[0]:
                    starting_row = i + 1
                    break
            if starting_row is not None:
                df = pd.read_csv(file_path, skiprows = starting_row)
                #renames columns accounting for either S11 or S22 measurements
                df.rename(columns={'Log Mag S11': 'dB', 'Log Mag S22': 'dB'}, inplace=True)
                
                df['Frequency(GHz)'] = df['Frequency(MHz)'] / 1000          
                #converts dB to linear so the values can be averaged
                df['Linear Values'] = 10**(df['dB']/10)
                            
        #handles alternative file format
        elif '! Anritsu VNA Master:' in first_line:
            for i, line in enumerate(file):
                if 'PNT' in line.split(',')[0]:
                    starting_row = i + 1
                    break
            if starting_row is not None:
                df = pd.read_csv(file_path, skiprows = starting_row)

                df = df.drop('PNT', axis=1)
                #for simplicity only uses trace 1 data (S11)
                df.rename(columns={'TR1.FREQ.GHZ': 'Frequency(GHz)', 'TR1.LOG MAG.dB': 'dB'}, inplace=True)
                df['Linear Values'] = 10**(df['dB']/10)

        num_groups = (len(df) + group_size - 1)// group_size
        #averages every x points in the data
        freq_averages = [df['Frequency(GHz)'].iloc[i * group_size:(i + 1) * group_size].mean() for i in range(num_groups)]
        linear_averages = [df['Linear Values'].iloc[i * group_size:(i + 1) * group_size].mean() for i in range(num_groups)]

        averages_df = pd.DataFrame({'Frequency(GHz)': freq_averages, 'Linear': linear_averages})
        averages_df['dB'] = 10*np.log10(averages_df['Linear'])
                
    return averages_df
                           
#generates a graph for each file
def generate_graphs(script_directory, full_csv_path):        
    points_to_avg = int(input('Enter the number of data points to average: '))
    #looks for every csv file in the file directory
    for csv_file in os.listdir(full_csv_path):
        if csv_file.endswith('.csv'):
            file_path = os.path.join(full_csv_path, csv_file)
            averages_df = retrieve_data(file_path, points_to_avg)

            #obtains the file name without the file path and '.csv' included
            sample = os.path.splitext(csv_file)[0]
            graphs_path = 'Graphs'

            #plots graph and saves it to graphs path
            plt.figure(figsize=(10, 6))
            plt.plot(averages_df['Frequency(GHz)'], averages_df['dB'], color='blue', linestyle='-', linewidth=2)
            plt.title(f'dB (10 Averages) - {sample}')
            plt.xlabel('Frequency(GHz)')
            plt.ylabel('dB')
            plt.grid(True)
            if not os.path.exists(f'{graphs_path}/{sample}_{points_to_avg}_pt_avg.png'):
                plt.savefig(f'{graphs_path}/{sample}_{points_to_avg}_pt_avg.png')
            plt.close()

    print(f'png files saved to {script_directory}\{graphs_path}.')          

#creates a table of the average dB levels for each sample across all frequencies
def average_dB_whole_range(script_directory, full_csv_path):
    points_to_avg = int(input('Enter the number of data points to average: '))
    columns = ['Sample', 'Average dB 1-12GHz']
    averaged_dB_df = pd.DataFrame(columns=columns)
    data_to_add = []
  
    #looks for every csv file in the file directory
    for csv_file in os.listdir(full_csv_path):
        if csv_file.endswith('.csv'):
            file_path = os.path.join(full_csv_path, csv_file)
            averages_df = retrieve_data(file_path, points_to_avg)

            #calculates the average dB
            lin_avg = averages_df['Linear'].mean()
            dB_avg = round(10*np.log10(lin_avg), 2)
            #gets the sample name
            sample = os.path.splitext(csv_file)[0]
            #adds the data to a list
            data_to_add.append({'Sample': sample, 'Average dB 1-12GHz': dB_avg})

    #adds each key value pair (sample and dB_avg) as a row in the table                                       
    for row_data in data_to_add:
        averaged_dB_df.loc[len(averaged_dB_df)] = row_data

    output_csv_title = 'output_average_dB_whole.csv'
    averaged_dB_df.to_csv(output_csv_title, index=False)
    print(f'{output_csv_title} file saved to {script_directory}.')
    return averaged_dB_df

#creates a table of the average dB levels for each sample across the selected frequency range
def average_dB_select_range(script_directory, full_csv_path, start_range, end_range):
    points_to_avg = int(input('Enter the number of data points to average: '))
    #checks if the user specified start and end frequencies are within the overall frequency range, and are the right way round
    if 1 <= start_range <= 12 and 1 <= end_range <= 12 and end_range > start_range:
        columns = ['Sample', f'Average dB {start_range}-{end_range}GHz']
        averaged_dB_df = pd.DataFrame(columns=columns)
        data_to_add = []

        #looks for every csv file in the file directory
        for csv_file in os.listdir(full_csv_path):
            if csv_file.endswith('.csv'):
                file_path = os.path.join(full_csv_path, csv_file)
                averages_df = retrieve_data(file_path, points_to_avg)
                #filters out data not in the specified range
                filtered_df = averages_df[(averages_df['Frequency(GHz)'] >= start_range) & (averages_df['Frequency(GHz)'] <= end_range)]

                #calculates the average dB
                lin_avg_in_range = filtered_df['Linear'].mean()
                dB_avg_in_range = round(10*np.log10(lin_avg_in_range), 2)
                #gets the sample name
                sample = os.path.splitext(csv_file)[0]
                #adds the data to a list
                data_to_add.append({'Sample': sample, f'Average dB {start_range}-{end_range}GHz': dB_avg_in_range})

        #adds each key value pair (sample and dB_avg) as a row in the table                                        
        for row_data in data_to_add:
            averaged_dB_df.loc[len(averaged_dB_df)] = row_data

        output_csv_title = f'output_average_dB_{start_range}-{end_range}GHz.csv'
        averaged_dB_df.to_csv(output_csv_title, index=False)
        print(f'{output_csv_title} file saved to {script_directory}.')
        return averaged_dB_df
        
    else:
        print('Inputted frequencies not in the range of 1-12GHz, or end frequency is less than start frequency.')

#creates a table of bandwidths under the specified dB threshold
def bandwidths_under_select_dB(script_directory, full_csv_path, dB_threshold):
    points_to_avg = int(input('Enter the number of data points to average: '))
    #creates the results table with NaN in the first row to prevent errors from concatenation of empty tables
    master_bandwidths_df = pd.DataFrame([[np.nan] * 4], columns=['Sample', 'Bandwidth(GHz)', 'Start Frequency(GHz)', 'End Frequency(GHz)'])

    #looks for every csv file in the file directory
    for csv_file in os.listdir(full_csv_path):
        if csv_file.endswith('.csv'):
            file_path = os.path.join(full_csv_path, csv_file)
            sample = os.path.splitext(csv_file)[0]
            averages_df = retrieve_data(file_path, points_to_avg)
            #filters out data where the dB level is above the threshold
            filtered_df = averages_df[averages_df['dB'] < dB_threshold]

            #creates a dictionary of the indexes and frequency column
            index_dict = dict(zip(filtered_df.index, filtered_df['Frequency(GHz)']))

            bandwidths = {}
            dicts_collection = []

            #uses the indexes to find consecutive frequency values and returns each set of consecutive values in a separate bandwidths dictionary
            for i in range(len(index_dict)):
                current_index = list(index_dict.keys())[i]
                next_index = list(index_dict.keys())[i + 1] if i < len(index_dict) - 1 else None

                if next_index is not None and next_index - current_index == 1:
                    bandwidths[current_index] = index_dict[current_index]
                else:
                    bandwidths[current_index] = index_dict[current_index]
                    dicts_collection.append(bandwidths.copy())
                    bandwidths = {}

            #initialises the bandwidths table
            columns = ['Sample', 'Bandwidth(GHz)', 'Start Frequency(GHz)', 'End Frequency(GHz)']
            bandwidths_df = pd.DataFrame(columns=columns)
            data_to_add = []

            #calculates the bandwidth for each set of values and adds the data to a list
            for bandwidth_array in dicts_collection:
                start_freq = list(bandwidth_array.values())[0]
                end_freq = list(bandwidth_array.values())[-1]
                bandwidth = end_freq - start_freq
                row_data = {'Sample': sample, 'Bandwidth(GHz)': round(bandwidth, 2), 'Start Frequency(GHz)': round(start_freq, 2), 'End Frequency(GHz)': round(end_freq, 2)}
                if bandwidth > 0:
                    data_to_add.append(row_data)
            #adds the data in the list to the table
            if data_to_add:
                bandwidths_df = pd.DataFrame(data_to_add)
                #joins the df for each sample to the master table
                master_bandwidths_df = pd.concat([master_bandwidths_df, bandwidths_df], ignore_index=True)

    #removes the NaN row
    master_bandwidths_df = master_bandwidths_df.dropna()
    #generates csv file
    output_csv_title = f'output_bandwidths_{dB_threshold}dB.csv'
    master_bandwidths_df.to_csv(output_csv_title, index=False)
    print(f'{output_csv_title} file saved to {script_directory}.')
    return master_bandwidths_df

#generates a scatter plot as a png file along with the data in csv format
def generate_average_plot(script_directory, full_csv_path):
    points_to_avg = int(input('Enter the number of data points to average: '))
    columns = ['Sample', 'Number of Layers', 'Average dB 1-12GHz']
    avg_dB_vs_layers_df = pd.DataFrame(columns=columns)
    data_to_add = []
    for csv_file in os.listdir(full_csv_path):
        if csv_file.endswith('.csv'):
            file_path = os.path.join(full_csv_path, csv_file)
            averages_df = retrieve_data(file_path, points_to_avg)
            
            #obtains the file name without the file path and '.csv' included
            sample = os.path.splitext(csv_file)[0]

            #finds the number of layers from the file name
            pattern = r'(\d+\.?\d*)x'
            matches = re.findall(pattern, sample)
            #converts matched values to float and calculate the sum
            no_of_layers = sum(float(match) for match in matches)

            #calculates the average dB
            lin_avg = averages_df['Linear'].mean()
            dB_avg = round(10*np.log10(lin_avg), 2)

            data_to_add.append({'Sample': sample, 'Number of Layers': no_of_layers, 'Average dB 1-12GHz': dB_avg})

    #adds each key value pair (sample and dB_avg) as a row in the table                                        
    for row_data in data_to_add:
        avg_dB_vs_layers_df.loc[len(avg_dB_vs_layers_df)] = row_data

    output_csv_title = 'output_scatter_plot_data.csv'
    avg_dB_vs_layers_df.to_csv(output_csv_title, index=False)
    print(f'{output_csv_title} file saved to {script_directory}.')
    
    graphs_path = 'Graphs'

    x = avg_dB_vs_layers_df['Number of Layers']
    y = avg_dB_vs_layers_df['Average dB 1-12GHz']
    n = avg_dB_vs_layers_df['Sample']
    
    #plots graph and displays it
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)
    plt.title('Average dB vs Number of Layers Scatter Plot')
    plt.xlabel('Number of Layers')
    plt.ylabel('Average dB')
    for i in range(len(avg_dB_vs_layers_df)):
        plt.text(x[i], y[i], n[i], fontsize = 7)
    plt.savefig(f'{graphs_path}/Average dB vs Number of Layers Scatter Plot.png')
    print(f'png file saved to {script_directory}\{graphs_path}.')
    plt.show()
    plt.close()

def generate_bandwidths_plot(script_directory, full_csv_path):
    points_to_avg = int(input('Enter the number of data points to average: '))
    #creates the results table with NaN in the first row to prevent errors from concatenation of empty tables
    columns = ['Sample', 'Number of Layers', 'Bandwidths under -5dB(GHz)']
    bandwidths_vs_layers_df = pd.DataFrame(columns=columns)
    data_to_add = []

    #looks for every csv file in the file directory
    for csv_file in os.listdir(full_csv_path):
        if csv_file.endswith('.csv'):
            file_path = os.path.join(full_csv_path, csv_file)
            sample = os.path.splitext(csv_file)[0]
            
            pattern = r'(\d+\.?\d*)x'
            matches = re.findall(pattern, sample)
            #converts matched values to float and calculate the sum
            no_of_layers = sum(float(match) for match in matches)
            
            averages_df = retrieve_data(file_path, points_to_avg)
            #filters out data where the dB level is above the threshold
            filtered_df = averages_df[averages_df['dB'] < -5]

            #creates a dictionary of the indexes and frequency column
            index_dict = dict(zip(filtered_df.index, filtered_df['Frequency(GHz)']))

            bandwidths = {}
            dicts_collection = []

            #uses the indexes to find consecutive frequency values and returns each set of consecutive values in a separate bandwidths dictionary
            for i in range(len(index_dict)):
                current_index = list(index_dict.keys())[i]
                next_index = list(index_dict.keys())[i + 1] if i < len(index_dict) - 1 else None

                if next_index is not None and next_index - current_index == 1:
                    bandwidths[current_index] = index_dict[current_index]
                else:
                    bandwidths[current_index] = index_dict[current_index]
                    dicts_collection.append(bandwidths.copy())
                    bandwidths = {}

            bandwidths_total = 0

            #calculates the bandwidth for each set of values and adds the data to a list
            for bandwidth_array in dicts_collection:
                start_freq = list(bandwidth_array.values())[0]
                end_freq = list(bandwidth_array.values())[-1]
                bandwidth = end_freq - start_freq
                bandwidths_total += bandwidth
            row_data = {'Sample': sample, 'Number of Layers': no_of_layers, 'Bandwidths under -5dB(GHz)': round(bandwidths_total, 2)}
            data_to_add.append(row_data)

def generate_minimum_plot(script_directory, full_csv_path):
    points_to_avg = int(input('Enter the number of data points to average: '))
    #creates the results table
    columns = ['Sample', 'Number of Layers', 'Frequency (GHz)', 'Minimum dB Level']
    min_vs_layers_df = pd.DataFrame(columns=columns)
    data_to_add = []

    #looks for every csv file in the file directory
    for csv_file in os.listdir(full_csv_path):
        if csv_file.endswith('.csv'):
            file_path = os.path.join(full_csv_path, csv_file)
            sample = os.path.splitext(csv_file)[0]
            
            pattern = r'(\d+\.?\d*)x'
            matches = re.findall(pattern, sample)
            #converts matched values to float and calculate the sum
            no_of_layers = sum(float(match) for match in matches)
            
            averages_df = retrieve_data(file_path, points_to_avg)
            
            min_linear = min(averages_df['Linear'])
            min_dB = round(10*np.log10(min_linear), 2)
            min_freq = float(averages_df.query(f'Linear=={min_linear}')['Frequency(GHz)'].iloc[0])

            data_to_add.append({'Sample': sample, 'Number of Layers': no_of_layers, 'Frequency (GHz)': min_freq, 'Minimum dB Level': min_dB})

    #adds each key value pair (sample and min) as a row in the table   
    for row_data in data_to_add:
        min_vs_layers_df.loc[len(min_vs_layers_df)] = row_data
        
    output_csv_title = 'output_min_plot_data.csv'
    min_vs_layers_df.to_csv(output_csv_title, index=False)
    print(f'{output_csv_title} file saved to {script_directory}.')
    
    graphs_path = 'Graphs'

    x = min_vs_layers_df['Frequency (GHz)']
    y = min_vs_layers_df['Minimum dB Level']
    n = min_vs_layers_df['Sample']
    
    #plots graph and displays it
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)
    plt.title('Minimum dB Level vs Frequency Scatter Plot')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Minimum dB Level')
    for i in range(len(min_vs_layers_df)):
        plt.text(x[i], y[i], n[i], fontsize = 7)
    plt.savefig(f'{graphs_path}/Minimum dB vs Frequency Scatter Plot.png')
    print(f'png file saved to {script_directory}\{graphs_path}.') 
    plt.show()
    plt.close()


frame_csv_directory = 'CSVs/'
vna_script_path = os.path.abspath(__file__)
#gets the directory containing the script
vna_script_directory = os.path.dirname(vna_script_path)
#creates the full path to the CSV directory
frame_csv_path = os.path.join(vna_script_directory, frame_csv_directory)

#welcome text
print('Welcome! Please view the options for analysis below.')
options = [
    {1: 'Generate dB vs frequency graphs'},
    {2: 'View average dB results for whole frequency range'},
    {3: 'View average dB results for a select frequency range'},
    {4: 'Find all bandwidths under a select dB level'},
    {5: 'Produce a scatter plot of average dB level vs number of layers'},
    {6: 'Produce a scatter plot of bandwidths under -5dB vs number of layers'},
    {7: 'Produce a scatter plot of the minimum dB level vs number of layers'}]

for option in options:
    for key, value in option.items():
        print(f'{key}: {value}')
        
selected_key = int(input('Enter the number corresponding to the option you want to select: '))
selected_option = next((option[selected_key] for option in options if selected_key in option), None)

if selected_option is not None:
    print(f'You selected: {selected_option}')
    if selected_key == 1:
        generate_graphs(vna_script_directory, frame_csv_path)
    if selected_key == 2:
        averaged_db_df = average_dB_whole_range(vna_script_directory, frame_csv_path)
        min_value = averaged_db_df['Average dB 1-12GHz'].min()
        min_index = averaged_db_df['Average dB 1-12GHz'].idxmin()
        sample_of_min = averaged_db_df.at[min_index, 'Sample']
        print(f'The sample with the lowest average dB is {sample_of_min} with an average dB level of {min_value}.') 
    if selected_key == 3:
        print('Please enter a range within 1-12GHz to see the average dB results for each sample.')
        input_start_range = float(input('Enter the start frequency value in GHz: '))
        input_end_range = float(input('Enter the end frequency value in GHz: '))
        averaged_db_df = average_dB_select_range(vna_script_directory, frame_csv_path, input_start_range, input_end_range)
        min_value = averaged_db_df[f'Average dB {input_start_range}-{input_end_range}GHz'].min()
        min_index = averaged_db_df[f'Average dB {input_start_range}-{input_end_range}GHz'].idxmin()
        sample_of_min = averaged_db_df.at[min_index, 'Sample']
        print(f'The sample with the lowest average dB is {sample_of_min} with an average dB level of {min_value}.')
    if selected_key == 4:
        input_dB_threshold = float(input('Please enter the dB level you would like to see bandwidths under: '))
        bandwidths_under_select_dB(vna_script_directory, frame_csv_path, input_dB_threshold)
    if selected_key == 5:
        generate_average_plot(vna_script_directory, frame_csv_path)
    if selected_key == 6:
        generate_bandwidths_plot(vna_script_directory, frame_csv_path)
    if selected_key == 7:
        generate_minimum_plot(vna_script_directory, frame_csv_path)
else:
    print('Invalid selection.')

print('Program finished.')



    

                  
