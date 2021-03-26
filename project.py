# ============================================================== #
#  SECTION: Imports                                              #
# ============================================================== #

# standard library

# third party library
import numpy as np
import pandas as pd

# local


# ============================================================== #
#  SECTION: Globals                                              #
# ============================================================== #

# CSV of motor vehicle collisions
MOTOR_VEHICLE_COLLISIONS_CSV = 'Motor_Vehicle_Collisions_-_Crashes.csv'
# cleaned CSV of motor vehicle collisions
CLEAN_MOTOR_VEHICLE_COLLISIONS_CSV = 'Clean_Motor_Vehicle_Collisions_-_Crashes.csv'
# sample data for testing
SAMPLE_DATA_CSV = 'sample_data.csv'

# optional controls
# flag to clean data
CLEAN_DATA = True
# flag to debug
DEBUG = True

# month you'd like data for
DESIRED_MONTH = '5'
# non covid year you'd like data for
NON_COVID_YEAR = '2020'
# covid year you'd like data for
COVID_YEAR = '2018'

# bin size for longitude
LONGITUDE_BIN = .2
# bin size for latitude
LATITUDE_BIN = .2

# unacceptable empty columns
BAD_NAN_COLUMNS = ['LATITUDE', 'LONGITUDE']

# mapping of words to their primary synonym - factors
LESSER_TO_PRIMARY_SYNONYM_FACTOR = {}
# mapping of primary word to its list of synonyms - factors
PRIMARY_TO_LESSER_SYNONYM_FACTOR = {}

# mapping of words to their primary synonym - vehicles
LESSER_TO_PRIMARY_SYNONYM_VEHICLE = {}
# mapping of primary word to its list of synonyms - vehicles
PRIMARY_TO_LESSER_SYNONYM_VEHICLE = {}


# ============================================================== #
#  SECTION: Class Definitions                                   #
# ============================================================== #


# ============================================================== #
#  SECTION: Helper Definitions                                   #
# ============================================================== #

def clean_data():
    """Clean data and save it to a new csv."""
    # read motor_vehicle_collision data into data frame
    collision_df = pd.read_csv(MOTOR_VEHICLE_COLLISIONS_CSV)

    # delete columns that we do not care about
    for column in ['COLLISION_ID']:
        del collision_df[column]

    # one hot code factor column_name mapped to column_values
    factor_columns = dict.fromkeys(list(PRIMARY_TO_LESSER_SYNONYM_FACTOR.keys()), [0] * len(collision_df))
    # one hot code vehicle column_name mapped to column_values
    vehicle_columns = dict.fromkeys(list(PRIMARY_TO_LESSER_SYNONYM_VEHICLE.keys()), [0] * len(collision_df))

    # TEMPORARY BLOCK FOR GATHERING UNIQUE IDS
    unique_factors = set()
    unique_vehicles = set()
    for row_index, row in collision_df.iterrows():
        for column_type_index in range(1, 6):
            # name of factor column
            factor_column_name = 'CONTRIBUTING FACTOR VEHICLE {}'.format(column_type_index)
            # name of vehicle type column
            vehicle_type_column_name = 'VEHICLE TYPE CODE {}'.format(column_type_index)
            unique_factors.add(row[factor_column_name])
            unique_vehicles.add(row[vehicle_type_column_name])
        # hopefully all unique vals will be gathered by this
        if row_index > 100000:
            print('dd')
            break
    print(unique_factors)
    print(unique_vehicles)
    return

    # set of indexes to drop
    dropped_indexes = []
    for row_index, row in collision_df.iterrows():
        # drop rows with Nan values in BAD_NAN_COLUMNS
        drop_flag = False
        for column in BAD_NAN_COLUMNS:
            if pd.isnull(row[column]):
                drop_flag = True
                dropped_indexes.append(row_index)
                break
        if drop_flag:
            continue

        # drop rows that are outside the dates we'd like to track
        month, day, year = row['CRASH DATE'].split('/')
        if year not in [NON_COVID_YEAR, COVID_YEAR] or month != DESIRED_MONTH:
            print('{}/{}'.format(month, year))
            dropped_indexes.append(row_index)
        else:
            # quantize latitude by rounding
            row['LATITUDE'] = round(row['LATITUDE'] / LATITUDE_BIN) * LATITUDE_BIN
            # quantize longitude by rounding
            row['LONGITUDE'] = round(row['LONGITUDE'] / LONGITUDE_BIN) * LONGITUDE_BIN
            # quantize crash time by flooring to the hour
            row['CRASH TIME'] = row['CRASH TIME'].split(':')[0]

            for column_type_index in range(1, 6):
                # name of factor column
                factor_column_name = 'CONTRIBUTING FACTOR VEHICLE {}'.format(column_type_index)
                # name of vehicle type column
                vehicle_type_column_name = 'VEHICLE TYPE CODE {}'.format(column_type_index)

                if not pd.isnull(row[factor_column_name]) and row[factor_column_name] != 'Unspecified':
                    factor_columns[LESSER_TO_PRIMARY_SYNONYM_FACTOR[row[factor_column_name]]][row_index] = 1

                if not pd.isnull(row[vehicle_type_column_name]) and row[vehicle_type_column_name] != 'Unspecified':
                    vehicle_columns[LESSER_TO_PRIMARY_SYNONYM_VEHICLE[row[vehicle_type_column_name]]][row_index] = 1

    # remove columns to be one hot coded
    for column_type_index in range(1, 6):
        # name of factor column
        factor_column_name = 'CONTRIBUTING FACTOR VEHICLE {}'.format(column_type_index)
        del collision_df[factor_column_name]
        # name of vehicle type column
        vehicle_type_column_name = 'VEHICLE TYPE CODE {}'.format(column_type_index)
        del collision_df[vehicle_type_column_name]
    # add columns to perform one hot coding
    for column_name in list(factor_columns.keys()):
        collision_df.insert(len(factor_columns.keys()), 'CONTRIBUTING FACTOR {}'.format(column_name), factor_columns[column_name])
    for column_name in list(vehicle_columns.keys()):
        collision_df.insert(len(vehicle_columns.keys()), 'CONTRIBUTING VEHICLE TYPE {}'.format(column_name), vehicle_columns[column_name])

    # drop invalid rows
    collision_df.drop(dropped_indexes, inplace=True)

    # save cleaned data into csv
    collision_df.to_csv(CLEAN_MOTOR_VEHICLE_COLLISIONS_CSV, index=False)

# ============================================================== #
#  SECTION: Main                                                 #
# ============================================================== #


if __name__ == '__main__':
    if CLEAN_DATA:
        clean_data()
