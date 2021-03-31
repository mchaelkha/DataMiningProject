# ============================================================== #
#  SECTION: Imports                                              #
# ============================================================== #

# standard library
from collections import defaultdict

# third party library
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


def read_data():
    collision_df = pd.read_csv(MOTOR_VEHICLE_COLLISIONS_CSV)
    columns = ['CRASH DATE', 'CRASH TIME', 'BOROUGH', 'LATITUDE', 'LONGITUDE', 'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS INJURED',
            'NUMBER OF PEDESTRIANS KILLED', 'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED']
    collision_df = collision_df[columns].copy()
    collision_df['CRASH DATE'] = pd.to_datetime(collision_df['CRASH DATE'])
    collision_df['CRASH TIME'] = pd.to_datetime(collision_df['CRASH TIME'], format='%H:%M')
    # filter out where borough, latitude, and longitude are NaN
    loc_df = collision_df.dropna(thresh=len(columns) - 2)
    return loc_df


def df_filter_by(df, prop, value):
    """
    Ways to filter the dataframe.
    """
    if prop == 'borough':
        return df[df['BOROUGH'] == value]
    elif prop == 'year':
        return df[df['CRASH DATE'].dt.year == value]
    elif prop == 'month':
        return df[df['CRASH DATE'].dt.month == value]
    elif prop == 'day':
        return df[df['CRASH DATE'].dt.day_name() == value]
    elif prop == 'hour':
        return df[df['CRASH TIME'].dt.hour == value]


def query_accidents_by_borough_and_year(loc_df):
    """
    Section 1
    """
    counts = defaultdict(list)
    for year in YEARS:
        # print('Showing year:', year)
        year_df = df_filter_by(loc_df, 'year', year)
        for borough in BOROUGH_COORDS.keys():
            borough_df = df_filter_by(year_df, 'borough', borough)
            # print('   ', borough, year_df.shape[0])
            counts[borough].append(borough_df.shape[0])
    return YEARS, counts


def query_accidents_by_month_and_year(loc_df):
    """
    Section 2
    """
    counts = defaultdict(list)
    for year in YEARS:
        # print('Showing year:', year)
        year_df = df_filter_by(loc_df, 'year', year)
        for month in MONTHS.keys():
            month_df = df_filter_by(year_df, 'month', month)
            # print('   ', MONTHS[month], month_df.shape[0])
            counts[MONTHS[month]].append(month_df.shape[0])
    return YEARS, counts


def query_accidents_by_weekday_and_year(loc_df):
    """
    Section 4
    """
    counts = defaultdict(list)
    for year in YEARS:
        # print('Showing year:', year)
        year_df = df_filter_by(loc_df, 'year', year)
        for day in DAYS:
            day_df = df_filter_by(year_df, 'day', day)
            # print('   ', day, day_df.shape[0])
            counts[day].append(day_df.shape[0])
    return YEARS, counts


def query_accidents_by_weekday_and_time_and_year(loc_df):
    """
    Section 5
    """
    counts = defaultdict(lambda: defaultdict(list))
    for year in YEARS:
        # print('Showing year:', year)
        year_df = df_filter_by(loc_df, 'year', year)
        for hour in HOURS:
            # print('   ', hour)
            # quantize times by top of the hour
            hour_df = df_filter_by(year_df, 'hour', hour)
            for day in DAYS:
                day_df = df_filter_by(hour_df, 'day', day)
                # print('       ', day, day_df.shape[0])
                counts[year][day].append(day_df.shape[0])
    return YEARS, counts


def query_accidents_by_borough_and_month_and_year(loc_df):
    """
    Section 6
    """
    counts = defaultdict(lambda: defaultdict(list))
    for year in YEARS:
        # print('Showing year:', year)
        year_df = df_filter_by(loc_df, 'year', year)
        for borough in BOROUGH_COORDS.keys():
            # print('   ', borough)
            borough_df = df_filter_by(year_df, 'borough', borough)
            for month in MONTHS.keys():
                month_df = df_filter_by(borough_df, 'month', month)
                # print('       ', MONTHS[month], month_df.shape[0])
                counts[year][borough].append(month_df.shape[0])
    return YEARS, counts


def query_accidents_by_borough_and_day_and_year(loc_df):
    """
    Section 6
    """
    counts = defaultdict(lambda: defaultdict(list))
    for year in YEARS:
        # print('Showing year:', year)
        year_df = df_filter_by(loc_df, 'year', year)
        for borough in BOROUGH_COORDS.keys():
            # print('   ', borough)
            borough_df = df_filter_by(year_df, 'borough', borough)
            for day in DAYS:
                day_df = df_filter_by(borough_df, 'day', day)
                # print('       ', day, day_df.shape[0])
                counts[year][borough].append(day_df.shape[0])
    return YEARS, counts


def query_accidents_by_borough_and_hour_and_year(loc_df):
    """
    Section 6
    """
    counts = defaultdict(lambda: defaultdict(list))
    for year in YEARS:
        # print('Showing year:', year)
        year_df = df_filter_by(loc_df, 'year', year)
        for borough in BOROUGH_COORDS.keys():
            # print('   ', borough)
            borough_df = df_filter_by(year_df, 'borough', borough)
            for hour in HOURS:
                # quantize times by top of the hour
                hour_df = df_filter_by(borough_df, 'hour', hour)
                # print('       ', hour, hour_df.shape[0])
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


def visualize_one(loc_df):
    print('Question 1')
    years, data = query_accidents_by_borough_and_year(loc_df)
    plot_multiple_bar_by_metric(data, years, title='Accidents by Borough from 2013 to 2020', xlabel='Year', ylabel='Accidents')


def visualize_two(loc_df):
    print('Question 2')
    years, data = query_accidents_by_month_and_year(loc_df)
    plot_multiple_bar_by_metric(data, years, title='Accidents by Month from 2013 to 2020', xlabel='Year', ylabel='Accidents')


def visualize_four(loc_df):
    print('Question 4')
    years, data = query_accidents_by_weekday_and_year(loc_df)
    plot_multiple_bar_by_metric(data, years, title='Accidents by Weekday from 2013 to 2020', xlabel='Year', ylabel='Accidents')


def visualize_five(loc_df, subplot=False):
    """
    Set subplot to True for side-by-side and similarly scaled plots. Easier for comparisons.
    Otherwise plot each year individually.
    """
    print('Question 5')
    years, data = query_accidents_by_weekday_and_time_and_year(loc_df)

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
    else:
        for year in years:
            plot_multiple_bar_by_metric(data[year], HOURS, title=f'Accidents by Hour each Weekday in {year}', xlabel='Hour', ylabel='Accidents')


def visualize_six(loc_df, subplot=False):
    """
    Set subplot to True for side-by-side and similarly scaled plots. Easier for comparisons.
    Otherwise plot each year individually.
    """
    print('Question 6')
    print('months')
    years, data = query_accidents_by_borough_and_month_and_year(loc_df)
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
    else:
        for year in years:
            plot_multiple_bar_by_metric(data[year], MONTHS.keys(), title=f'Accidents in Boroughs by Month in {year}', xlabel='Month', ylabel='Accidents')

    print('weekdays')
    years, data = query_accidents_by_borough_and_day_and_year(loc_df)
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
    else:
        for year in years:
            plot_multiple_bar_by_metric(data[year], DAYS, title=f'Accidents in Boroughs by Weekday in {year}', xlabel='Weekday', ylabel='Accidents')

    print('hours')
    years, data = query_accidents_by_borough_and_hour_and_year(loc_df)
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
    else:
        for year in years:
            plot_multiple_bar_by_metric(data[year], HOURS, title=f'Accidents in Boroughs by Hour in {year}', xlabel='Hour', ylabel='Accidents')



# ============================================================== #
#  SECTION: Main                                                 #
# ============================================================== #


if __name__ == '__main__':
    loc_df = read_data()
    visualize_five(loc_df, subplot=True)
    visualize_six(loc_df, subplot=True)
