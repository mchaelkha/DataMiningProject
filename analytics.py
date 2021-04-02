# ============================================================== #
#  SECTION: Imports                                              #
# ============================================================== #

# standard library
from collections import defaultdict

# third party library
from geopy import distance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local


# ============================================================== #
#  SECTION: Globals                                              #
# ============================================================== #

# CSV of motor vehicle collisions
MOTOR_VEHICLE_COLLISIONS_CSV = 'Motor_Vehicle_Collisions_-_Crashes.csv'

# ============================================================== #
#  SECTION: Class Definitions                                   #
# ============================================================== #


# ============================================================== #
#  SECTION: Helper Definitions                                   #
# ============================================================== #

BOROUGH_COORDS = {
    'STATEN ISLAND': [40.58, 74.15],
    'BRONX': [40.84, 73.86],
    'QUEENS': [40.73, 73.79],
    'MANHATTAN': [40.78, 73.97],
    'BROOKLYN': [40.68, 73.94]
}

LATITUDE_BOUNDS = [40.4, 41]

LONGITUDE_BOUNDS = [-74.4, -73.5]

YEARS = range(2013, 2021)

MONTHS = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}

DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

HOURS = range(0, 24)


def df_filter_by(df, prop, value):
    """
    Ways to filter the dataframe.
    """
    if prop == 'borough':
        return df[df['BOROUGH'] == value]
    elif prop == 'year':
        return df[df['CRASH TIME'].dt.year == value]
    elif prop == 'month':
        return df[df['CRASH TIME'].dt.month == value]
    elif prop == 'day':
        return df[df['CRASH TIME'].dt.day_name() == value]
    elif prop == 'hour':
        return df[df['CRASH TIME'].dt.hour == value]


def read_data(set_location=False, save_cleaned=False, save_years=False):
    """
    Read and clean the data for most use cases.
    Additional cleaning needed for type of collision and quantizing values.
    """
    collision_df = pd.read_csv(MOTOR_VEHICLE_COLLISIONS_CSV)
    # columns to keep
    columns = ['CRASH DATE', 'CRASH TIME', 'BOROUGH', 'LATITUDE', 'LONGITUDE', 'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS INJURED',
            'NUMBER OF PEDESTRIANS KILLED', 'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED']
    collision_df = collision_df[columns].copy()

    # Combine crash date and crash time into datetime
    collision_df['CRASH TIME'] = pd.to_datetime(collision_df['CRASH DATE'] + ' ' + collision_df['CRASH TIME'], format='%m/%d/%Y %H:%M')
    del collision_df['CRASH DATE']

    # fill NaN lat and long with 0 to treat 0 and NaN the same
    collision_df.fillna({'LATITUDE': 0, 'LONGITUDE': 0}, inplace=True)
    cleaned_df = collision_df.copy()

    # filter out where borough is blank or coordinates are 0
    cleaned_df = cleaned_df[cleaned_df['BOROUGH'].notnull() | cleaned_df['LONGITUDE'] != 0]

    # clean an edge case on Queensboro Bridge having wrong longitude and no borough
    cleaned_df.loc[cleaned_df['LONGITUDE'] == -201.23706, ['BOROUGH', 'LONGITUDE']] = ['MANHATTAN', -73.95337]

    # arbitrarily set the coordinates for records that have a borough, but do not have a location (1160 records)
    if set_location:
        for borough in BOROUGH_COORDS.keys():
            location = BOROUGH_COORDS[borough]
            cleaned_df.loc[((cleaned_df['BOROUGH'] == borough) & (cleaned_df['LATITUDE'] == 0)), ['LATITUDE', 'LONGITUDE']] = location

    # filter out coordinates that are not in NYC
    min_lat, max_lat = LATITUDE_BOUNDS
    min_long, max_long = LONGITUDE_BOUNDS
    within_lat = (min_lat < cleaned_df['LATITUDE']) & (cleaned_df['LATITUDE'] < max_lat)
    within_long = (min_long < cleaned_df['LONGITUDE']) & (cleaned_df['LONGITUDE'] < max_long)
    # these records should have borough labels and should not be thrown out as we handled that before
    no_lat_long = ((cleaned_df['LATITUDE'] == 0) & (cleaned_df['LATITUDE'] == 0))
    cleaned_df = cleaned_df[(within_lat & within_long) | no_lat_long]

    # save and output
    year_dfs = {}
    for year in YEARS:
        year_df = df_filter_by(cleaned_df, 'year', year)
        if save_years:
            year_df.to_csv(f'year_{year}.csv')
        year_dfs[year] = year_df
    if save_cleaned:
        cleaned_df.to_csv('cleaned_analytics_data.csv')
    return cleaned_df, year_dfs


def within_bounds(lat, long, other_lat, other_long, radius):
    """
    TODO: implement using geopy distance
    """


def query_accidents_by_borough_and_year(cleaned_df, print_step=False):
    """
    Section 1
    """
    counts = defaultdict(list)
    for year in YEARS:
        if print_step:
            print('Showing year:', year)
        year_df = df_filter_by(cleaned_df, 'year', year)
        for borough in BOROUGH_COORDS.keys():
            borough_df = df_filter_by(year_df, 'borough', borough)
            if print_step:
                print('   ', borough, year_df.shape[0])
            counts[borough].append(borough_df.shape[0])
    return YEARS, counts


def query_accidents_by_month_and_year(cleaned_df, print_step=False):
    """
    Section 2
    """
    counts = defaultdict(list)
    for year in YEARS:
        if print_step:
            print('Showing year:', year)
        year_df = df_filter_by(cleaned_df, 'year', year)
        for month in MONTHS.keys():
            month_df = df_filter_by(year_df, 'month', month)
            if print_step:
                print('   ', MONTHS[month], month_df.shape[0])
            counts[MONTHS[month]].append(month_df.shape[0])
    return YEARS, counts


def query_accidents_by_weekday_and_year(cleaned_df, print_step=False):
    """
    Section 4
    """
    counts = defaultdict(list)
    for year in YEARS:
        if print_step:
            print('Showing year:', year)
        year_df = df_filter_by(cleaned_df, 'year', year)
        for day in DAYS:
            day_df = df_filter_by(year_df, 'day', day)
            if print_step:
                print('   ', day, day_df.shape[0])
            counts[day].append(day_df.shape[0])
    return YEARS, counts


def query_accidents_by_weekday_and_time_and_year(cleaned_df, print_step=False):
    """
    Section 5
    """
    counts = defaultdict(lambda: defaultdict(list))
    for year in YEARS:
        if print_step:
            print('Showing year:', year)
        year_df = df_filter_by(cleaned_df, 'year', year)
        for hour in HOURS:
            if print_step:
                print('   ', hour)
            # quantize times by top of the hour
            hour_df = df_filter_by(year_df, 'hour', hour)
            for day in DAYS:
                day_df = df_filter_by(hour_df, 'day', day)
                if print_step:
                    print('       ', day, day_df.shape[0])
                counts[year][day].append(day_df.shape[0])
    return YEARS, counts


def query_accidents_by_borough_and_month_and_year(cleaned_df,print_step=False):
    """
    Section 6
    """
    counts = defaultdict(lambda: defaultdict(list))
    for year in YEARS:
        if print_step:
            print('Showing year:', year)
        year_df = df_filter_by(cleaned_df, 'year', year)
        for borough in BOROUGH_COORDS.keys():
            if print_step:
                print('   ', borough)
            borough_df = df_filter_by(year_df, 'borough', borough)
            for month in MONTHS.keys():
                month_df = df_filter_by(borough_df, 'month', month)
                if print_step:
                    print('       ', MONTHS[month], month_df.shape[0])
                counts[year][borough].append(month_df.shape[0])
    return YEARS, counts


def query_accidents_by_borough_and_day_and_year(cleaned_df, print_step=False):
    """
    Section 6
    """
    counts = defaultdict(lambda: defaultdict(list))
    for year in YEARS:
        if print_step:
            print('Showing year:', year)
        year_df = df_filter_by(cleaned_df, 'year', year)
        for borough in BOROUGH_COORDS.keys():
            if print_step:
                print('   ', borough)
            borough_df = df_filter_by(year_df, 'borough', borough)
            for day in DAYS:
                day_df = df_filter_by(borough_df, 'day', day)
                if print_step:
                    print('       ', day, day_df.shape[0])
                counts[year][borough].append(day_df.shape[0])
    return YEARS, counts


def query_accidents_by_borough_and_hour_and_year(cleaned_df, print_step=False):
    """
    Section 6
    """
    counts = defaultdict(lambda: defaultdict(list))
    for year in YEARS:
        if print_step:
            print('Showing year:', year)
        year_df = df_filter_by(cleaned_df, 'year', year)
        for borough in BOROUGH_COORDS.keys():
            if print_step:
                print('   ', borough)
            borough_df = df_filter_by(year_df, 'borough', borough)
            for hour in HOURS:
                # quantize times by top of the hour
                hour_df = df_filter_by(borough_df, 'hour', hour)
                if print_step:
                    print('       ', hour, hour_df.shape[0])
                counts[year][borough].append(hour_df.shape[0])
    return YEARS, counts


def plot_multiple_bar_by_metric(data, metric, title='', xlabel='', ylabel='', colors=None, total_width=0.8, single_width=1, legend=True):
    """
    Dynamically generate multiple bar graph/histogram
    """
    _, ax = plt.subplots()
    # default bar and legend colors
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors.append(u'slateblue')
        colors.append(u'lightgreen')
    n_bars = len(data)
    bar_width = total_width / n_bars
    # legend bars
    bars = []
    for i, (_, values) in enumerate(data.items()):
        # where to reposition bar from x
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # x and y position for each bar
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        bars.append(bar[0])
    # add title, labels, legend
    if legend:
        ax.legend(bars, data.keys())
    plt.xticks(range(len(metric)), metric)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def subplot_multiple_bar_by_metric(ax, data, metric, title='', xlabel='', ylabel='', y_max=0, y_scale=None, total_width=0.8, single_width=1, legend=True):
    """
    Dynamically generate multiple bar graph/histogram
    """
    # default bar and legend colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors.append(u'slateblue')
    colors.append(u'lightgreen')
    n_bars = len(data)
    bar_width = total_width / n_bars
    # legend bars
    bars = []
    for i, (_, values) in enumerate(data.items()):
        # where to reposition bar from x
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # x and y position for each bar
        for x, y in enumerate(values):
            y_max = max(y_max, y)
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        bars.append(bar[0])
    # add title, labels, legend
    if legend:
        ax.legend(bars, data.keys())
    plt.xticks(range(len(metric)), metric)
    if y_scale is not None:
        plt.yticks(np.arange(0, y_max, y_scale))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def visualize_one(cleaned_df):
    print('Question 1')
    years, data = query_accidents_by_borough_and_year(cleaned_df)
    plot_multiple_bar_by_metric(data, years, title='Accidents by Borough from 2013 to 2020', xlabel='Year', ylabel='Accidents')


def visualize_two(cleaned_df):
    print('Question 2')
    years, data = query_accidents_by_month_and_year(cleaned_df)
    plot_multiple_bar_by_metric(data, years, title='Accidents by Month from 2013 to 2020', xlabel='Year', ylabel='Accidents')


def visualize_four(cleaned_df):
    print('Question 4')
    years, data = query_accidents_by_weekday_and_year(cleaned_df)
    plot_multiple_bar_by_metric(data, years, title='Accidents by Weekday from 2013 to 2020', xlabel='Year', ylabel='Accidents')


def visualize_five(cleaned_df, subplot=False):
    """
    Set subplot to True for side-by-side and similarly scaled plots. Easier for comparisons.
    Otherwise plot each year individually.
    """
    print('Question 5')
    years, data = query_accidents_by_weekday_and_time_and_year(cleaned_df)

    if subplot:
        fig = plt.figure()
        for index, year in enumerate(years[:4]):
            legend = index == 0
            ax = fig.add_subplot(2, 2, index + 1)
            subplot_multiple_bar_by_metric(ax, data[year], HOURS, title=f'Accidents in Boroughs by Hour in {year}', xlabel='Hour', ylabel='Accidents', y_max=2800, y_scale=500, legend=legend)
        plt.show()

        fig = plt.figure()
        for index, year in enumerate(years[4:]):
            legend = index == 0
            ax = fig.add_subplot(2, 2, index + 1)
            subplot_multiple_bar_by_metric(ax, data[year], HOURS, title=f'Accidents in Boroughs by Hour in {year}', xlabel='Hour', ylabel='Accidents', y_max=2800, y_scale=500, legend=legend)
        plt.show()
    for year in years:
        plot_multiple_bar_by_metric(data[year], HOURS, title=f'Accidents by Hour each Weekday in {year}', xlabel='Hour', ylabel='Accidents')


def visualize_six(cleaned_df, month=True, weekday=True, hour=True, subplot=False):
    """
    Set subplot to True for side-by-side and similarly scaled plots. Easier for comparisons.
    Otherwise plot each year individually.
    """
    print('Question 6')
    if month:
        print('months')
        years, data = query_accidents_by_borough_and_month_and_year(cleaned_df)
        if subplot:
            fig = plt.figure()
            for index, year in enumerate(years[:4]):
                legend = index == 0
                ax = fig.add_subplot(2, 2, index + 1)
                subplot_multiple_bar_by_metric(ax, data[year], MONTHS.keys(), title=f'Accidents in Boroughs by Month in {year}', xlabel='Month', ylabel='Accidents', y_max=5000, y_scale=1000, legend=legend)
            plt.show()

            fig = plt.figure()
            for index, year in enumerate(years[4:]):
                legend = index == 0
                ax = fig.add_subplot(2, 2, index + 1)
                subplot_multiple_bar_by_metric(ax, data[year], MONTHS.keys(), title=f'Accidents in Boroughs by Month in {year}', xlabel='Month', ylabel='Accidents', y_max=5000, y_scale=1000, legend=legend)
            plt.show()
        for year in years:
            plot_multiple_bar_by_metric(data[year], MONTHS.keys(), title=f'Accidents in Boroughs by Month in {year}', xlabel='Month', ylabel='Accidents')

    if weekday:
        print('weekdays')
        years, data = query_accidents_by_borough_and_day_and_year(cleaned_df)
        if subplot:
            fig = plt.figure()
            for index, year in enumerate(years[:4]):
                legend = index == 0
                ax = fig.add_subplot(2, 2, index + 1)
                subplot_multiple_bar_by_metric(ax, data[year], DAYS, title=f'Accidents in Boroughs by Weekday in {year}', xlabel='Weekday', ylabel='Accidents', y_max=7900, y_scale=1000, legend=legend)
            plt.show()

            fig = plt.figure()
            for index, year in enumerate(years[4:]):
                legend = index == 0
                ax = fig.add_subplot(2, 2, index + 1)
                subplot_multiple_bar_by_metric(ax, data[year], DAYS, title=f'Accidents in Boroughs by Weekday in {year}', xlabel='Weekday', ylabel='Accidents', y_max=7900, y_scale=1000, legend=legend)
            plt.show()
        for year in years:
            plot_multiple_bar_by_metric(data[year], DAYS, title=f'Accidents in Boroughs by Weekday in {year}', xlabel='Weekday', ylabel='Accidents')

    if hour:
        print('hours')
        years, data = query_accidents_by_borough_and_hour_and_year(cleaned_df)
        if subplot:
            fig = plt.figure()
            for index, year in enumerate(years[:4]):
                legend = index == 0
                ax = fig.add_subplot(2, 2, index + 1)
                subplot_multiple_bar_by_metric(ax, data[year], HOURS, title=f'Accidents in Boroughs by Hour in {year}', xlabel='Hour', ylabel='Accidents', y_max=4000, y_scale=500, legend=legend)
            plt.show()

            fig = plt.figure()
            for index, year in enumerate(years[4:]):
                legend = index == 0
                ax = fig.add_subplot(2, 2, index + 1)
                subplot_multiple_bar_by_metric(ax, data[year], HOURS, title=f'Accidents in Boroughs by Hour in {year}', xlabel='Hour', ylabel='Accidents', y_max=4000, y_scale=500, legend=legend)
            plt.show()
        for year in years:
            plot_multiple_bar_by_metric(data[year], HOURS, title=f'Accidents in Boroughs by Hour in {year}', xlabel='Hour', ylabel='Accidents')


def run_visualizations(cleaned_df, subplot=False):
    visualize_one(cleaned_df)
    visualize_two(cleaned_df)
    visualize_four(cleaned_df)
    visualize_five(cleaned_df, subplot=subplot)
    visualize_six(cleaned_df, subplot=subplot)


# ============================================================== #
#  SECTION: Main                                                 #
# ============================================================== #


if __name__ == '__main__':
    cleaned_df, years_df = read_data()
    run_visualizations(cleaned_df, subplot=True)
