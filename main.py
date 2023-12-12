import statistics as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os
import time
import re


def get_data_types(df):
    dtypes = df.dtypes
    data_types = []
    for idx, data_type in enumerate(dtypes):
        if data_type == 'object':
            data_types.append('string')
        elif data_type == 'int64' and df.iloc[:, idx].isin([0, 1]).all():
            data_types.append('boolean')
        else:
            data_types.append(data_type.name)
    return data_types


def get_data_range(df):
    data_range = []
    for col in df:
        if len(set(df[col])) <= 10:
            data_range.append(set(df[col]))
        else:
            data_range.append("[ " + str(min(df[col])) + " - " + str(max(df[col])) + " ]")
    return data_range


def get_data_outlier(df):
    outlier_data = []
    for col in df:
        if df[col].dtype != 'object' and not (df[col].dtype == 'int64' and df[col].isin([0, 1]).all()):
            q1, q3 = np.percentile(sorted(df[col]), [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)

            outlier_data.append([x for x in df[col] if x <= lower_bound or x >= upper_bound])
        else:
            outlier_data.append(None)
    return outlier_data


def get_box_plot(df, addr):
    for col in df:
        if df[col].dtype != 'object' and not (df[col].dtype == 'int64' and df[col].isin([0, 1]).all()):
            plt.boxplot(df[col])
            plt.title(col)
            plt.savefig(addr + col + "BoxPlot.png")
            plt.show()


def analysis(df, base):
    for col in df.columns:
        if col != base:
            if df[col].dtype == 'object' or (df[col].dtype == 'int64' and df[col].isin([0, 1]).all()):
                cross_tab = pd.crosstab(df[base], df[col])
                max_0 = cross_tab.iloc[0].idxmax()
                max_0_count = cross_tab.loc[0, max_0]
                min_0 = cross_tab.iloc[0].idxmin()
                min_0_count = cross_tab.loc[0, min_0]
                max_1 = cross_tab.iloc[1].idxmax()
                max_1_count = cross_tab.loc[1, max_1]
                min_1 = cross_tab.iloc[1].idxmin()
                min_1_count = cross_tab.loc[1, min_1]
                print("{} with the highest count of 0: {} (Count: {})".format(col, max_0, max_0_count))
                print("{} with the highest count of 1: {} (Count: {})".format(col, max_1, max_1_count))
                print("{} with the lowest count of 0: {} (Count: {})".format(col, min_0, min_0_count))
                print("{} with the lowest count of 1: {} (Count: {})".format(col, min_1, min_1_count))
            else:
                correlation, pv = scipy.stats.pearsonr(df[base], df[col])
                print("Correlation between {} and {}: {}".format(col, base, correlation))
                print("and the p-value: {}".format(pv))


def part1(main_data_frame, addr):
    results = pd.DataFrame({'Name': [], 'Type': [], 'Range': [],
                            'Max': [], 'Min': [], 'Mean': [],
                            'Mode': [], 'Median': [], 'Outlier': []})
    results['Name'] = main_data_frame.columns
    results['Type'] = get_data_types(main_data_frame)
    results['Range'] = get_data_range(main_data_frame)
    results['Max'] = [max(main_data_frame[col]) for col in results['Name']]
    results['Min'] = [min(main_data_frame[col]) for col in results['Name']]
    results['Mean'] = [round(sum(main_data_frame[col]) / len(main_data_frame[col]), 5)
                       if results['Type'][idx] != "string" else None for idx, col in enumerate(results['Name'])]
    results['Mode'] = [st.mode(main_data_frame[col]) for col in results['Name']]
    results['Median'] = [st.median(main_data_frame[col]) for col in results['Name']]
    results['Outlier'] = get_data_outlier(main_data_frame)
    results.to_csv(addr + 'part1.csv')
    get_box_plot(main_data_frame, addr)
    # analysis(main_data_frame, 'Heart Attack Risk')


def accuracy_data(df):
    syntax_accuracy = []
    for col in df.columns:
        i = 0
        f = 0
        s = 0
        for d in df[col]:
            try:
                int(d)
                i += 1
            except:
                try:
                    float(d)
                    f += 1
                except:
                    s += 1
        syntax_accuracy.append(max(i, f, s) / len(df[col]))
    return syntax_accuracy


def validity_data(df):
    valid = []
    # Patient ID RegEx
    valid_id = re.compile(r'\w{7}$')
    valid.append(df['Patient ID'].str.match(valid_id).mean())
    # Sex RegEx
    valid_sex = re.compile(r'^Male$|^Female$')
    valid.append(df['Sex'].str.match(valid_sex).mean())
    # Diet RegEx
    valid_diet = re.compile(r'^Average$|^Unhealthy$|^Healthy$')
    valid.append(df['Diet'].str.match(valid_diet).mean())
    # Country RegEx
    valid_country = re.compile(r'^[A-Z][\sa-zA-Z]{3,}$')
    valid.append(df['Country'].str.match(valid_country).mean())
    # Continent RegEx
    valid_continent = re.compile(r'^[A-Z][\sa-zA-Z]{3,}$')
    valid.append(df['Continent'].str.match(valid_continent).mean())
    # Hemisphere RegEx
    valid_hemisphere = re.compile(r'^(Nor|Sou)thern Hemisphere$')
    valid.append(df['Hemisphere'].str.match(valid_hemisphere).mean())
    validity = []
    i = 0
    for col in df.columns:
        if col in ['Patient ID', 'Sex', 'Diet', 'Country', 'Continent', 'Hemisphere']:
            validity.append(round(valid[i], 5))
            i += 1
        else:
            is_valid = 0
            for data in df[col]:
                try:
                    int(data)
                    is_valid += 1
                except:
                    try:
                        float(data)
                        is_valid += 1
                    except:
                        continue
            validity.append(round(is_valid / len(df[col]), 5))
    return validity


def check_consistency(df):
    p = [[] for _ in df.columns]
    age_range = 0.8 * max(df['Age']) + 0.2 * min(df['Age'])
    ex_range = 0.8 * max(df['Exercise Hours Per Week']) + 0.2 * min(df['Exercise Hours Per Week'])
    sn_range = 0.8 * max(df['Sedentary Hours Per Day']) + 0.2 * min(df['Sedentary Hours Per Day'])
    bmi_max_range = 0.8 * max(df['BMI']) + 0.2 * min(df['BMI'])
    bmi_min_range = 0.2 * max(df['BMI']) + 0.8 * min(df['BMI'])
    true_place = [('Argentina', 'South America', 'Southern Hemisphere'),
                  ('Australia', 'Australia', 'Southern Hemisphere'),
                  ('Brazil', 'South America', 'Southern Hemisphere'),
                  ('Canada', 'North America', 'Northern Hemisphere'),
                  ('China', 'Asia', 'Northern Hemisphere'),
                  ('Colombia', 'South America', 'Northern Hemisphere'),
                  ('France', 'Europe', 'Northern Hemisphere'),
                  ('Germany', 'Europe', 'Northern Hemisphere'),
                  ('India', 'Asia', 'Northern Hemisphere'),
                  ('Italy', 'Europe', 'Northern Hemisphere'),
                  ('Japan', 'Asia', 'Northern Hemisphere'),
                  ('New Zealand', 'Australia', 'Southern Hemisphere'),
                  ('Nigeria', 'Africa', 'Northern Hemisphere'),
                  ('South Africa', 'Africa', 'Southern Hemisphere'),
                  ('South Korea', 'Asia', 'Northern Hemisphere'),
                  ('Spain', 'Europe', 'Northern Hemisphere'),
                  ('Thailand', 'Asia', 'Northern Hemisphere'),
                  ('United Kingdom', 'Europe', 'Northern Hemisphere'),
                  ('United States', 'North America', 'Northern Hemisphere'),
                  ('Vietnam', 'Asia', 'Northern Hemisphere')]
    for index, row in df.iterrows():
        if row['Age'] > age_range and \
                row['Exercise Hours Per Week'] > ex_range and \
                row['Sedentary Hours Per Day'] > sn_range:
            p[df.columns.get_loc('Age')].append(index)
        if (row['BMI'] > bmi_max_range and row['Obesity'] == 0) or (row['BMI'] < bmi_min_range and row['Obesity'] == 1):
            p[df.columns.get_loc('BMI')].append(index)
        if (row['Country'], row['Continent'], row['Hemisphere']) not in true_place:
            p[df.columns.get_loc('Country')].append(index)
    return p


def part2(main_data_frame, addr):
    results = pd.DataFrame({'Name': [], 'Count': [], 'Null': [],
                            'Accuracy': [], 'Completeness': [], 'Validity': [],
                            'Currentness': [], 'Consistency': []})
    results['Name'] = main_data_frame.columns
    results['Count'] = [len(main_data_frame[col]) for col in main_data_frame.columns]
    results['Null'] = main_data_frame.isnull().sum().tolist()
    results['Accuracy'] = accuracy_data(main_data_frame)
    results['Completeness'] = [1 - round((empty / results['Count'][idx]), 5)
                               for idx, empty in enumerate(main_data_frame.isnull().sum())]
    results['Validity'] = validity_data(main_data_frame)
    last_change_time = max(os.path.getmtime('.\\Data\\main.csv'),
                           os.path.getctime('.\\Data\\main.csv'),
                           os.path.getatime('.\\Data\\main.csv'))
    results['Currentness'] = [time.ctime(last_change_time) for _ in main_data_frame.columns]
    results['Consistency'] = check_consistency(main_data_frame)
    results.to_csv(addr + 'part2.csv')


def missing_data_handler(df):
    drop = []
    for col in df.columns:
        if df[col].isna().sum() > 1000:
            drop.append(col)
    for col in drop:
        df = df.drop(col, axis=1)
    df.fillna(df.median(numeric_only=True).round(1), inplace=True)
    string_columns = df.select_dtypes(include=['object']).columns
    df[string_columns] = df[string_columns].fillna(df[string_columns].mode().iloc[0])
    return df


def normalization(df):
    for col in df.columns:
        if df[col].dtype == 'float64' or (df[col].dtype == 'int64' and not df[col].isin([0, 1]).all()):
            df[col] = round(df[col] / df[col].max(), 5)
    return df


def outlier_removal(df):
    for col in df:
        if df[col].dtype != 'object' and not (df[col].dtype == 'int64' and df[col].isin([0, 1]).all()):
            q1, q3 = np.percentile(sorted(df[col]), [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            for i, d in enumerate(df[col]):
                if d < lower_bound or d > upper_bound:
                    df = df.drop(i)
    return df


def new_features(df):
    df['Ex-Sd hours per week'] = df['Exercise Hours Per Week'] - 7 * df['Sedentary Hours Per Day']
    df = df.drop(columns='Exercise Hours Per Week', axis=1).drop(columns='Sedentary Hours Per Day', axis=1)
    df['Addiction'] = df.apply(lambda row: 0 if row['Smoking'] + row['Alcohol Consumption'] == 0 else 1, axis=1)
    df = df.drop(columns='Smoking', axis=1).drop(columns='Alcohol Consumption', axis=1)
    col = df.pop('Heart Attack Risk')
    df.insert(len(df.columns), 'Heart Attack Risk', col)
    return df


def draw(df, addr):
    for col in df.columns:
        if df[col].dtype != 'object' and col != 'Heart Attack Risk':
            if df[col].isin([0, 1]).all():
                continue
                binary_pairs = [str(int(a)) + str(int(b)) for a, b in zip(df[col], df['Heart Attack Risk'])]
                bins = ['00', '01', '10', '11']
                freq = [binary_pairs.count(bin) for bin in bins]
                plt.figure(figsize=(15, 10))
                plt.bar(['no ' + col + '-no Risk', 'no ' + col + '-Risk', col + '-no Risk', col + '-Risk'], freq)
                plt.xlabel('Bins')
                plt.ylabel('Frequency')
                plt.title('Histogram of binary pairs')
                plt.savefig(addr + col + "Histogram.png")
                plt.show()
            else:
                no_HAR = []
                for i, d in enumerate(df[col]):
                    if df['Heart Attack Risk'][i] == 0:
                        no_HAR.append(d)
                plt.hist(no_HAR, bins=10, edgecolor='black')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.title(col + ' no Heart Attack Risk Distribution')
                plt.savefig(addr + col + "-no-HAR-Histogram.png")
                plt.show()
                HAR = []
                for i, d in enumerate(df[col]):
                    if df['Heart Attack Risk'][i] == 1:
                        HAR.append(d)
                plt.hist(HAR, bins=10, edgecolor='black')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.title(col + ' Heart Attack Risk Distribution')
                plt.savefig(addr + col + "-HAR-Histogram.png")
                plt.show()
        else:
            if col != 'Patient ID':
                pairs = [str(a) + str(int(b)) for a, b in zip(df[col], df['Heart Attack Risk'])]
                pair = list(dict.fromkeys(pairs))
                freq = [pairs.count(p) for p in pair]
                plt.figure(figsize=(15, 10))
                plt.bar(pair, freq)
                plt.xlabel('Bins')
                plt.ylabel('Frequency')
                plt.title('Histogram of binary pairs')
                plt.savefig(addr + col + "Histogram.png")
                plt.show()

    return


def part3(main_data_frame, addr):
    modified_df = missing_data_handler(main_data_frame)
    modified_df = new_features(modified_df)
    modified_df = outlier_removal(modified_df)
    # data reduction
    modified_df = modified_df.drop(columns=['Hemisphere', 'Continent', 'Obesity'], axis=1)
    # illustration
    draw(modified_df, addr)
    modified_df = normalization(modified_df)
    modified_df.to_csv(addr + 'part3.csv')


if __name__ == '__main__':
    main_data = pd.read_csv('.\\Data\\main.csv')
    main_data[['Systolic', 'Diastolic']] = main_data['Blood Pressure'].str.split('/', expand=True)
    main_data['Systolic'] = main_data['Systolic'].astype('int64')
    main_data['Diastolic'] = main_data['Diastolic'].astype('int64')
    main_data = main_data.drop(columns='Blood Pressure', axis='columns')
    col = main_data.pop('Heart Attack Risk')
    main_data.insert(len(main_data.columns), 'Heart Attack Risk', col)
    print("----------------------------PART1------------------------------")
    part1(main_data, '.\\Results\\Part1\\')
    print("----------------------------PART2------------------------------")
    part2(main_data, '.\\Results\\Part2\\')
    print("----------------------------PART3------------------------------")
    part3(main_data, '.\\Results\\Part3\\')
