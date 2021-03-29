# ============================================================== #
#  SECTION: Imports                                              #
# ============================================================== #

# standard library
from datetime import datetime

# third party library
import numpy as np
import pandas as pd
import progressbar

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

YEARS = ['2020', '2019']
MONTHS = ['03', '04']

LATITUDE_BIN_SIZE = .001
LONGITUDE_BIN_SIZE = .001

# factors that are generalized to failure to obeying traffic laws
FAILURE_TO_OBEY_TRAFFIC_FACTORS = {'Unsafe Lane Changing', 'Backing Unsafely',
                                   'Failure to Yield Right-of-Way', 'Traffic Control Disregarded',
                                   'Following Too Closely', 'Unsafe Speed', 'Turning Improperly',
                                   'Passing or Lane Usage Improper', 'Passing Too Closely',
                                   'Failure to Keep Right'}
# factors that are generalized to be a result of environmental conditions
ENVIRONMENTAL_FACTORS = {'View Obstructed/Limited', 'Animals Action', 'Steering Failure',
                         'Accelerator Defective', 'Brakes Defective', 'Pavement Slippery',
                         'Tinted Windows', 'Other Vehicular', 'Obstruction/Debris', 'Oversized Vehicle',
                         'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion', 'Passenger Distraction',
                         'Glare', 'Tire Failure/Inadequate', 'Tow Hitch Defective', 'Headlights Defective',
                         'Other Lighting Defects', 'Pavement Defective', 'Driverless/Runaway Vehicle',
                         'Outside Car Distraction', 'Windshield Inadequate',
                         'Traffic Control Device Improper/Non-Working', 'Reaction to Other Uninvolved Vehicle',
                         'Reaction to Uninvolved Vehicle', 'Lane Marking Improper/Inadequate'}
# factors that are generalized to be personal implications that are non drug related
PERSONAL_NON_DRUG_FACTORS = {'Driver Inattention/Distraction', 'Driver Inexperience', 'Lost Consciousness',
                                     'Fell Asleep', 'Illnes', 'Aggressive Driving/Road Rage',
                                     'Using On Board Navigation Device', 'Cell Phone (hand-Held)',
                                     'Listening/Using Headphones', 'Physical Disability', 'Illness',
                                     'Other Electronic Device', 'Shoulders Defective/Improper',
                                     'Eating or Drinking', 'Texting', 'Vehicle Vandalism', 'Cell Phone (hand-held)',
                                     'Cell Phone (hands-free)'}
# factors that are generalized to be drug related in nature
PERSONAL_DRUG_RELATED_FACTORS = {'Alcohol Involvement', 'Fatigued/Drowsy', 'Drugs (illegal)',
                                 'Aggressive Driving/Road Rage', 'Prescription Medication', 'Drugs (Illegal)'}

# mapping of column_names to sets of causation
GENERALIZED_CAUSE_TO_SPECIFIC = {'Failure To Obey Traffic Factor': FAILURE_TO_OBEY_TRAFFIC_FACTORS,
                                 'Environmental Cause Factor': ENVIRONMENTAL_FACTORS,
                                 'Personal Factor': PERSONAL_NON_DRUG_FACTORS,
                                 'Drug Related Factor': PERSONAL_DRUG_RELATED_FACTORS}

# non-vehicles
VEHICLE = {'tractor truck gasoline', 'mta truck', 'fdny truck', 'firetruck', 'food truck', 'beverage truck',
           'usps truck', 'pick-up truck', 'mailtruck', 'van/truck', 'ems truck', 'mack truck', 'box truck',
           'tow truck', 'truck', 'firtruck', 'dump truck', 'tow truck / wrecker', 'truck van', 'towe truck',
           'inte truck', 'dot truck', 'mail truck', 'truck flat', 'fd truck', 'truck trai', 'pick truck',
           'tractor truck diesel', 'armored truck', 'ups truck', 'van u', 'van s', 'van ford', 'marked van', 'van`',
           'van/r', 'vanette', 'rv/van', 'school van', 'moving van', 'vant', 'van (', 'van f', 'cargo van', 'van/b',
           'van/t', 'vanet', 'van w', 'usps van', 'van/', 'van t', 'van e', 'mini van', 'work van', 'van/transi',
           'refrigerated van', 'ems/van', 'van camper', 'dept van #', 'postal van', 'vehicle 2', 'livery vehicle',
           'station wagon/sport utility vehicle', 'farm vehicle', 'all-terrain vehicle', 'multi-wheeled vehicle',
           'passenger vehicle', 'ambul', 'ambulace', 'white ambu', 'ambulette', 'ambu', 'ambulance`', 'ambulence',
           'fdny ambul', 'ambulance', 'gen  ambul', 'buss', 'bus m', 'bus', 'short bus', 'mta bus  4', 'blu bus',
           'schoolbus', 'school bus', 'bus y', 'postal bus', 'omnibus', 'nyc bus', 'tour bus', 'livery bus',
           'pickup tru', 'pickup tow', 'pickup with mounted camper',
           'tow', 'e tow', 'tower', 'tow trick', 'towma', 'tow-t', 'towe', 'light towe', 'tow t', 'tow r', 'g tow',
           'towtr', 'towin', 'tow trk', 'car', 'postal car', 'smart car', 'tr/c', 'sanit', 'emergancy', 'miniv', 'sem',
           'pick-', 'tan p', 'amubulance', 'contr', 'tktr', 'kw tr', 'armored tr', 'fd tr', 'am/tr', 'ltr', 'tr/tr',
           'rv/tr', 'utility tr', 'pick-up tr', 'eletr', 'flatbed tr', 'tr', 'nyc firetr', 'tractor tr', 'tl tr',
           'pick up tr', 'freight tr', 'quality tr', 'fork-', 'e amb', 'ï¿½mbu', 'transit va', 'small com veh(4 tires) ',
           'vn', 'uhaul truc', 'train', 'jetski', 'ems a', 'buldozer', 'ems b', 'abulance', 'vms', 'motorbike', 'track',
           'impal', 't/ cr', 'truc', 'trk', 'e- motor b', 'sport utility / station wagon', 'moped elec', '5x8 t', 'r/v',
           'quad', 'mo pe', 'pedicab', 'cargo truc', 'usps truc,', 'ladder tru', 'firet truc', 'pumper tru',
           'rental tru', 'constructi', 'refuse tru', 'plow  truc', 'con ed tru', 'u haul tru', 'u-tru', 'semi-trail',
           'semi', 'semi-', 'semi trail', 'dirt bike', 'ebike', 'e-bik', 'e bik', 'e-bike', 'e bike', 'moped scoo',
           'moped/scoo','nyc s', 'nyc acs va', 'nyc sanita', 'nyc b', 'nyc a', 'nyc d', 'nyc', 'nycta', 'nyc ems',
           'nyc-s', 'nyc m', 'nyc fire t', 'nyc transi', 'nyc fd', 'nyc dot', 'nycha', 'nyc g', 'mopet', 'moped', 'mopen', 'bmw moped',
           'subn - amb', 'leased amb', 'g amb', 'amb', 'almbulance', 'motor home', 'motor', 'motorcycle', 'motorizeds',
           'motorized home', 'amulance', 'fire engin', 'fdny fire', 'fdny firet', 'firet', 'ups m', 'ups t',
           'fdny emt', 'nys a', 'nyfd', 'fd ny', 'nypd signa', 'nypd', 'dsny', 'fdny engin', 'nyu s', 'fdny #226',
           'fdny ems v', 'ny ems', 'nynj rail', 'fdny chief', 'fdny ems', 'dumpt', 'tuck',
           'large com veh(6 or more tires)', 'vehic', 'police veh', 'veh l', 'postal veh', 'snow plow', 'snow plowe',
           'plow', 'transit', 'transport', 'com trans', 'trans', 'u-hau', 'trc', 'red t', 'dhl t', 'mta t',
           'flat bed t', 'delivery t', 'com t', 'gas t', 'vms t', 'dilevery t', 'mac t', 'fre t', 'oil t', 'red t', 'del t',
           'esu t', 'unk t', 'car t', 'forklift t', 'box t', 'llv mail t', 'nat grid t', '52? t', 'dot t', 'gmc t', 'ems t',
           'track exca', 'ems h', 'ems', 'passenger', 'pkup', 'coupe', 'pick up', 'mobile', 'mobile foo', 'snowmobile','mobil',
           '3door', '3-door', '4door', 'taxi', '2 doo', 'doosk', '3 doo', 'hwy c', 'utilt', 'utility', 'utili',
           'utility ve', 'util wh', 'utility wh', 'sport utility / station wagon', 'emergency', 'uspst', 'fusion',
           'usps2', 'usps 88716', 'usps #7530', 'usps#', 'us postal', 'usps posta', 'us govt ve', 'tkp', 'tk', 'unknown ve',
           'electric m', 'vol', 'minicycle', 'cab', 'pedi cab', 'yellow cab''mta v', 'tlc v', 'dlev', 'dot v', 'freig delv',
           'dei v', 'suv', 'glp050vxev', 'pas v', 'deliv', 'delv', 'vav', 'p/v', 'gmc v', 'sprinter v', 'com v',
           'utv', 'conv', 'deiv', 'gov v', 'rv', 'dep v', 'rmp v', 'llv', 'lmtv', 'chevy expr', 'federal ex','fedex',
           'expre', 'fed ex', '2 whe', '3 whe', '3-whe', 'jeep', 'e pas', 'pass', 'g pas', 'pass-', 'pas 5', 'passe', 'pas', 'passa', 'e-mot',
           'moter', 'forkl', 'fork lift', 'forklift', 'lift','folk lift', 'golf cart', 'cart','ford', 'ford sprin',
           'wh ford co', 'ford econo', 'subn whi', 'sub', 'subn-', 'subn/', 'subur', 'subr', 'suburban', '4 axe', '8x20',
           'utyli','amubl', 'amula', 'ct', 'suv /', 'tt', '4dr', 'nv ca', 'john', 'chevrolet', 'mercedes', "gov't",
           'picku','flat', 'flat  bed', 'flat bed', 'flat/', 'flat rack', 'flatbed fr', 'flatb', 'lawn mower', 'lawnmower',
           'wagon', 'lunch wagon', 'e.m.s', 'schoo', '2 hor', 'u-hal', 'cms-t', 'anbul', 'sanitation', 'dlvr',
           'sanitaton', 'santa', 'sanitaion', 'santi', 'sanat', 'john deere', 'limou', 'limo', 'limo/', 'usps self',
           'self', 'self-', 'self insur', 'golf kart', 'golf', '2dr', '12 fe', '11 pa', '2ton', '4sedn', '4 run',
           '16m', '4 dr sedan', '2 ton', '15 pa', "12' o", '18 weeler', '600aj', '2- to', '12 pa', '315 e', '3dc-',
           '4dsd', '18 wh', '250-3', '38ab-', '2 dr sedan', '2 dr', '4ds', '11-va', 'e- bi', 'convertible',
           't650', 'f150xl pic', 'f550', 'f350', 'f650', 'f-250', 'mechanical', 'utll', 'fleet', 'rescu', 'esu rescue',
           'e450', 'mac 1', 'c 1', 'cat 4', 'e250', 'cat 9', 'ec2', 'cb534', 'nv150', 'c-1', 'kp160', 'jcb40', 't880',
           'e350', 'cat 3', 'e-350', 'bobcat 216', 'delie', 'del', 'delv.', 'delvi', 'delvr', 'delivery', 'boat',
           'toyot', 'toyota', 'wesco', 'excavator', 'escavator', 'geico', 'haul for h', 'uhaul', 'i-haul', 'u-haul',
           'go kart', 'e revel sc', 'e-scooter', 'gas scoote', 'e scooter', 'escoo', 'motor scoo', 'revel scoo',
           'e-sooter','r/v c', 'engi', 'fortl', 'com/a', 'red m', 'fed e', 'sudan', 'fedx', 'e sco', 'city of ne',
           'nissa', 'ice c', 'icecr', 'street swe', 'road sweep', 'sweep', 'sweeper', 'tank wh', 'tank', 'tanke',
           'hino tank', 'tanker', 'scl', 'g1`', 'smart', 'tour', 'livery omn', 'horse carr', 'horse carr', 'ram',
           'pick', 'cherry pic', 'pick rd', 'chevy', 'cherr', 'fuel', 'ice cream', 'whit', 'gray', 'red,', 'green', 'black',
           'hearse', 'white', 'blue', 'amabu', 'yamah', 'amazon spr', 'boom crane', 'lift boom', 'boom lift', 'booml', 'boom',
           'mail', 'grail', 'dodge', 'u.s. posta', 'work', 'sedan', 'cmixer', 'chvey', 'freig', 'repai', 'e-sco',
           'e com', 'east', 'e one', 'enclosed body - nonremovable enclosure', 'engin', 'ecom', 'e - b', 'enclosed body - removable enclosure',
           'econo', 'esu rep', 'econoline', 'epo', 'emt', 'ecoli', 'emrgn', 'enclo', 'cross', 'light', 'front', 'army',
           'boat', 'government', 'yello', 'commerical', 'sterl', 'mini', 'srf', 'sprin', 'c7c', 'camp', 'campe', 'atv p',
           'ram promas', 'silve', 'g sem', 'p/se', 'seagr', 'diese', 'segwa', 'postal ser', 'sen', 'sea', 'servi', 'mopad',
           'omt/t', 'roro', 'bsd', 'crwzk', 'trial', 'trt', 'pallet', 'comm trk', 'yll p', 'sm yw', 'humme', '?omme', 'bobca',
           'cat.', 'bobcat','bob cat', 'caterpilla', 'cat', 'catapillar', 'cate', 'catip', 'cater', 'cat p', 'cat32',
           'ulili', 'bobby', 'stree', 'pay loader', 'street cle', 'fed','comm.',
           'off r', 'glnen', 'speac', 'palfinger', 'kubot', 'psd', 'uliti', 'sfi', 'nttrl', 'mac f', 'sciss',
           'fd fi', 'com.', 'greeb', 'gr hs', 'e-scoter', 'elecr', 'elect', 'elec', 'elec. unic', 'electronic',
           'us ma', 'cushm', 'us go', 'us', 'usps/govt', 'uspos', 'u.s.p', 'usps', 'waste', 'food', 'comm food', 'uhal',
           'govt','gover', 'govt.', 'carry all', 'carri', 'cargo', 'me/be', 'oml/', 'd/v wb', 'ge/sc', 'refg/',
           'tr/ki', 'p/sh', 'p/u', 'mi/fu', 's/sp', 'no/bu', 'rd/s', 'safet''hd to', 'vas', 'conve', 'mta b', 'tcr',
           'skid-', 'gene', 'gas s', 'junst', 'pavin', 'prks', 'scava', 'backh', 'glben', 'piggy back', 'amdu',
           'laund', 'ringo', 'ltrl', 'freih', 'refq', 'tlc p', 'maxim', 'mack', 'feder', 'parke', 'bkhoe',
           'hoe-l', 'scomm', 'dept', 'g com', 'workh', 'big r', 'uk', 'dark color', 'movin', 'vab', 'bucke', 'bed',
           'man b', 'revel', 'speci', 'special co', 'tlr', 'sbn', 'orion', 'platf', 'hino', 'ynk',
           'yps', 'stake or rack', 'new y', 'state', 'sybn', 'hilow', 'trcic', 'g spc', 'lcomm', 'coach', 'man l',
           'backhoe lo', 'special pu', 'snow', 'shcoo', 'perm', 'trlr plt,', 'high', 'dolly', 'oms', 'armor', 'comm',
           'bucketload', 'gmc', 'posto', 'spinter va', 'sgws', 'mecha', 'qmz', 'hi ta', 'milli', 'liber', 'command po',
           'omni', 'naa', 'ukn', 'block', 'tandu', 'dot equipm', 'small', 'ariel', 'yellowpowe', 'mtr s', 'spec-', 'astro',
           'heavy', 'psr', 'ns am', 'ryder', 'navig', 'tree cutte', 'wineb', 'liebh', 'access a r', 'free', 'axo',
           'skid loade', 'g  co', 'oz mo', 'unlno', 'pois', 'hook', 'trl', 'swt', 'grumm', 'commu', 'box h', 'spc',
           'inter', 'olc', 'front-load', 'johnd', 'appor', 'mo-pe','mtriz', 'tcn', 'gator', 'burg', 'highl', 'alumi',
           'stack', 'sprinter', 'frieg', 'psp', 'vpg', 'lit direct', 'yale', 'dlr', 'skywatch', 'comix', 'tlc',
           'rolli', 'genie', 'heil', 'omnib', 'harve', 'box m', 'psh', 'token', 'stake', 'trc m', 'palle', 'renta',
           'fr`', 'dunba', 'feig', 'potal', 'paylo', 'olm', 'hand', 'aeria', 'trim', 'box p', 'rubbe', 'spec', 'mark',
           'frt', 'comb', 'tilla', 'cadet', 'uber', 'vend', 'depar', 'freight fl', 'bob c', 'app c', 'back hoe', 'dirtb',
           '\x7fomm', 'detac', 'tilt tande',
           'open body', 'app', 'refr', 'pedic', 'rep', 'frh', 'vms sign', 'dema-', 'const', 'lsa', 'loade', 'metal', 'lull',
           'spc p', 'hi-lo', 'oml', 'attac', 'sierra', 'refg', 'isuzu', 'polic', 'dot #', 'a-one', 'pch', 'omr', 'range',
           'chart', 'g omr', 'lumbe', 'stak', 'dual', 'box', 'internatio', 'skidsteerl', 'backhoe', 'rgs', 'tugge', 'priva',
           'rented boo', 'bulld', 'cont', 'hdc', 'aspha', 'liver', 'tcm', 'mcy b', 'ref g', 'hi lo',
           'freightlin', 'dot r', 'intl', 'con e', 'lma', 'rood', 'pallet jac', 'compa', 'lawn', 'vam', 'post', 'mta', 'fre',
           'winne', 'cont-', 'qbe i', 'vespa', 'workm', 'vendor cha', 'com', 'scom', 'message si', 'club', 'omt', 'fllet',
           'fd la', 'aport', 'mta u', 'jlg m', 'deagr', 'skid', 'tir', 'g psd', 'btm',
           'fltrl', 'suret', 'conti', 'power ladd', 'trlpm', 'itas', 'yw po'}

OBSTACLE = {'drone', 'cemen', 'freight', 'semitraile', 'post offic', 'salt', 'ladder 34',
            'fdny ladde', 'well driller', 'dump', 'dumpster', 'dumpe', 'trash', 'tr-trailer', 'trailer', 'ttrailer',
            'horse trai', 'tram', 'tra/r', 'camper tra', 'car traile', 'tractor', 'uhaul trai', 'frht trail', 'tract',
            'trailor', 'trai', 'industrial', 'house', 'house on w', 'us po', 'usps mail', 'garbage or refuse',
            'garba', 'garab', 'cabin', 'excav', 'power shov', 'uhaul box', 'food cart',
            'concrete m', 'broom', 'porta', 'refrg', 'slingshot', 'roads', 'pumpe', 'pump', 'pumper', 'jlg l', 'city',
            'drill rig', 'unattached', 'dirt-', 'can', 'dirt', 'scaff', 'trlr', 'tailg', 'ttailer', 'crane', 'offic',
            'grain', 'churc', 'brown', 'bobct', 'hrse', 'barri', 'bump', 'hosre', 'case', 'concr', 'whbl', 'trialer',
            'bulk agriculture', 'wheel barr', 'power','glass rack', 'historical', 'rd bldng m', 'art m','refri',
            'const equi', 'ladde', 'peter', 'city mta b', 'objec', 'ladd', 'livestock rack', 'road', 'hotdo'}

PEDESTRIAN = {'bicycle', 'scooter', 'push scoot', 'motorscooter', 'motorscoot',
               'scoot', 'razor scoo', 'escooter', 'scoo', 'courier',
               'couri', 'skateboard', 'e-skateboa', 'e skate bo', 'e-ska', 'skate', 'push', 'electric s', 'razor',
               'mot s', 'sc', 'foot', 'uni', 'scotter', 'rolle'}

UNKNOWN = {'unk.', 'unkown','wanc','uknown' ,'unknow', 'unkl', 'unkow', 'unkwn', 'unk l', 'unkn', 'unk/l', 'unknw', 'unknown', 'unk,',
           'unkno', 'unk', 'gov', '.','d', 'e', 'a', 'hd', 'c2', 'nd', 'pc', 'yw', 'bs', 'bk', '1c', 'ss',
           'dp', 'ps', 'sd', 'tl', 'wc', '1s', 'pz', 'tf', 'tc', 'sw', 'st', 'mp', 'ap', 'pk', 'd3', 'l1', 'es',
           'wg', 'j1', 'd1', 'na', 'e3', 'db', 'i1', 'c1', 'h1', 'ud', 'h3', 'e1', 'mc', 'ut', '4d', 'pu', 'yy', 'cm',
           'ip', 'co', 'c3', 'lp', 'left the s', '26 ft', '35 ft',
           'uknown', 'none', 'other', 'not i', 'short', 'foor', 'frht', 'n?a', 'unnko', 'm/a',
           'none'}

# mapping of column_names to sets of object types
GENERALIZED_TYPE_TO_SPECIFIC = {'OBSTACLE': OBSTACLE,
                                'PEDESTRIAN': PEDESTRIAN}


# ============================================================== #
#  SECTION: Class Definitions                                   #
# ============================================================== #


# ============================================================== #
#  SECTION: Helper Definitions                                   #
# ============================================================== #

def quantize_time(time):
    """Quantize time via flooring"""
    return time.split(':')[0]


def quantize_numeric(value, bin_size, function=round):
    """Quantize value via function"""
    return function(value / bin_size) * bin_size

# ============================================================== #
#  SECTION: Main                                                 #
# ============================================================== #


if __name__ == '__main__':
    # read motor_vehicle_collision data into data frame
    print('Reading CSV into dataframe ')
    collision_df = pd.read_csv(MOTOR_VEHICLE_COLLISIONS_CSV)
    print('Complete!\n')

    # remove columns of little value
    useless_columns = ['LOCATION', 'ON STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME', 'ZIP CODE', 'COLLISION_ID']
    for column in useless_columns:
        del collision_df[column]

    complete_columns = ['CRASH DATE', 'BOROUGH', 'LATITUDE', 'LONGITUDE']
    print('Dropping invalid rows')
    # initialize progress bar
    bar = progressbar.ProgressBar(maxval=len(collision_df),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    # set of indexes to drop
    date_dropped_indexes = []
    incomplete_dropped_indexes = []
    bar.start()
    for row_index, row in collision_df.iterrows():
        # get the crash date
        month, day, year = row['CRASH DATE'].split('/')
        # mark rows that do not satisfy date requirements
        if MONTHS and month not in MONTHS:
            date_dropped_indexes.append(row_index)
            # update progress bar
            bar.update(row_index + 1)
            continue
        if YEARS and year not in YEARS:
            date_dropped_indexes.append(row_index)
            # update progress bar
            bar.update(row_index + 1)
            continue

        # mark rows that do not satisfy completeness requirements
        for column in complete_columns:
            if pd.isnull(row[column]):
                incomplete_dropped_indexes.append(row_index)
                break

        # update progress bar
        bar.update(row_index + 1)
    # drop marked rows
    collision_df.drop(date_dropped_indexes + incomplete_dropped_indexes, inplace=True)

    # print statistics
    print('\n\mDropped {} rows for out of range dates'.format(len(date_dropped_indexes)))
    print('Dropped {} rows for incomplete data'.format(len(incomplete_dropped_indexes)))
    print('{} rows remaining'.format(len(collision_df)))
    print('Complete!\n')

    # reset dataframe indexing to sequential order
    collision_df.reset_index(drop=True, inplace=True)

    # quantize data
    print('Quantizing data')
    # initialize progress bar
    bar = progressbar.ProgressBar(maxval=len(collision_df),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    unique_boroughs = set()
    new_crash_time_column = []
    new_latitude_column = []
    new_longitude_column = []
    for row_index, row in collision_df.iterrows():
        # quantize time
        new_crash_time_column.append(quantize_time(row['CRASH TIME']))

        # quantize latitude
        new_latitude_column.append(quantize_numeric(row['LATITUDE'], bin_size=LATITUDE_BIN_SIZE, function=round))
        # quantize longitude
        new_longitude_column.append(quantize_numeric(row['LONGITUDE'], bin_size=LONGITUDE_BIN_SIZE, function=round))

        # update progress bar
        bar.update(row_index + 1)

        # grab unique boroughs for later step
        unique_boroughs.add(row['BOROUGH'])

    # replace old columns with new ones
    del collision_df['CRASH TIME']
    collision_df.insert(len(collision_df.columns), 'CRASH TIME', new_crash_time_column)
    del collision_df['LATITUDE']
    collision_df.insert(len(collision_df.columns), 'LATITUDE', new_latitude_column)
    del collision_df['LONGITUDE']
    collision_df.insert(len(collision_df.columns), 'LONGITUDE', new_longitude_column)
    print('\nComplete!\n')

    print('Ulterior Step Complete!')
    print('{} unique boroughs found'.format(len(unique_boroughs)))
    print('Unique boroughs: {}\n'.format(unique_boroughs))

    # one hot coding data
    print('One hot coding data')
    # initialize progress bar
    bar = progressbar.ProgressBar(maxval=len(collision_df),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # initializing one hot coding resulting columns
    unique_borough_columns = {borough: [0] * len(collision_df) for borough in unique_boroughs}
    unique_factor_columns = {factor: [0] * len(collision_df) for factor in list(GENERALIZED_CAUSE_TO_SPECIFIC.keys())}
    vehicle_type_columns = {vehicle_type: [0] * len(collision_df) for vehicle_type in list(GENERALIZED_TYPE_TO_SPECIFIC.keys())}

    year_columns = {year: [0] * len(collision_df) for year in YEARS}
    month_columns = {month: [0] * len(collision_df) for month in MONTHS}
    day_columns = {day: [0] * len(collision_df) for day in ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                                                            'Thursday', 'Friday', 'Saturday']}

    for row_index, row in collision_df.iterrows():
        # set values for boroughs one hot coding results
        unique_borough_columns[row['BOROUGH']][row_index] = 1

        # get crash date information
        crash_date = row['CRASH DATE']
        month, day, year = row['CRASH DATE'].split('/')
        year_columns[year][row_index] = 1
        month_columns[month][row_index] = 1
        day_columns[datetime.strptime(crash_date, '%m/%d/%Y').strftime('%A')][row_index] = 1

        unique_factors = set()
        for column_type_index in range(1, 6):
            # name of factor column
            factor_column_name = 'CONTRIBUTING FACTOR VEHICLE {}'.format(column_type_index)
            factor = row[factor_column_name]
            if pd.isnull(factor):
                break
            if factor != 'Unspecified' and not str.isnumeric(factor):
                for column_name in list(GENERALIZED_CAUSE_TO_SPECIFIC.keys()):
                    if factor in GENERALIZED_CAUSE_TO_SPECIFIC[column_name]:
                        unique_factor_columns[column_name][row_index] = 1
                        break

        unique_factors = set()
        for column_type_index in range(1, 6):
            # name of vehicle type column
            vehicle_type_column_name = 'VEHICLE TYPE CODE {}'.format(column_type_index)
            v_type = row[vehicle_type_column_name]
            if pd.isnull(v_type):
                break
            # even if a vehicle is not cited if a pedestrian is injured a pedestrian was involved
            for pedestrian_column in ['NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
                                      'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED']:
                if row[pedestrian_column]:
                    vehicle_type_columns['PEDESTRIAN'][row_index] = 1
                    break
            if v_type not in UNKNOWN and v_type not in VEHICLE:
                for column_name in list(GENERALIZED_TYPE_TO_SPECIFIC.keys()):
                    if v_type in GENERALIZED_TYPE_TO_SPECIFIC[column_name]:
                        vehicle_type_columns[column_name][row_index] = 1
                        break

        # update progress bar
        bar.update(row_index + 1)
    # add new generated columns
    for column_name in list(unique_borough_columns.keys()):
        collision_df.insert(len(unique_borough_columns.keys()), 'BOROUGH OF {}'.format(column_name),
                            unique_borough_columns[column_name])
    for column_name in list(vehicle_type_columns.keys()):
        collision_df.insert(len(vehicle_type_columns.keys()), 'INVOLVED TYPE {}'.format(column_name),
                            vehicle_type_columns[column_name])
    for column_name in list(unique_factor_columns.keys()):
        collision_df.insert(len(unique_factor_columns.keys()), column_name,
                            unique_factor_columns[column_name])
    for column_name in list(day_columns.keys()):
        collision_df.insert(0, 'CRASH WEEKDAY {}'.format(column_name),
                            day_columns[column_name])
    for column_name in list(month_columns.keys()):
        collision_df.insert(0, 'CRASH MONTH {}'.format(column_name),
                            month_columns[column_name])
    for column_name in list(year_columns.keys()):
        collision_df.insert(0, 'CRASH YEAR {}'.format(column_name),
                            year_columns[column_name])
    # delete now depreciated columns
    for column_type_index in range(1, 6):
        # name of factor column
        factor_column_name = 'CONTRIBUTING FACTOR VEHICLE {}'.format(column_type_index)
        del collision_df[factor_column_name]
    for column_type_index in range(1, 6):
        # name of factor column
        factor_column_name = 'VEHICLE TYPE CODE {}'.format(column_type_index)
        del collision_df[factor_column_name]
    del collision_df['CRASH DATE']
    del collision_df['BOROUGH']
    print('Complete!')

    # save cleaned data into csv
    collision_df.to_csv(CLEAN_MOTOR_VEHICLE_COLLISIONS_CSV, index=False)
