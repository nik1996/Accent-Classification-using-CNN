import pandas as pd
import sys
from sklearn.model_selection import train_test_split

def data_filter(df):
    '''
    Function to filter audio files based on df columns
    df column options: [age,age_onset,birth_place,filename,native_language,sex,speakerid,country]
    :param df (DataFrame): Full unfiltered DataFrame
    :return (DataFrame): Filtered DataFrame
    '''

    # Example to filter arabic and english and limit arabic to 70 audio files and english to 100 audio files.
    arabic = df[df['native_language'] == 'arabic'][:70]
    english = df[df.native_language == 'english'][:100]

    df = english.append(arabic)
    df = df[['age','filename','native_language','sex','speakerid','country']]
    df.to_csv('filtered_data.csv')

    return df

def df_split(df,test_size=0.2):
    '''
    Create train test split of DataFrame
    :param df (DataFrame): Pandas DataFrame of audio files to be split
    :param test_size (float): Percentage of total files to be split into test
    :return X_train, X_test, y_train, y_test (tuple): Xs are list of df['filename'] and Ys are df['native_language']
    '''
    return train_test_split(df['filename'],df['native_language'],test_size=test_size,random_state=1234)


if __name__ == '__main__':
    '''
    Console command example:
    python bio_data.csv
    '''
    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)
    filtered_df = data_filter(df)
    print (df_split(filtered_df))
