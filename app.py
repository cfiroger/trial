import dash
from dash import html, dcc, Output, Input, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash_loading_spinners as dls
import pandas as pd
import io
import numpy as np
from flask import Flask
import base64
import re
import math
from dash.exceptions import PreventUpdate
from math import cos, sin, atan2, sqrt, degrees, radians, asin
from plotly.subplots import make_subplots
import gc
global df_static
global xpol_report
global range_of_slider
global legend_html  #formates the legend, 5 columns below figure
global html_style  #colors legend always updated with legend_html by def legend_plotter
global df_cellsites


def distance_bearing(db_lat1, db_lat2, db_lon1, db_lon2, d_b_calc):
    try:
        db_lat1 = radians(db_lat1)
        db_lat2 = radians(db_lat2)
        db_lon1 = radians(db_lon1)
        db_lon2 = radians(db_lon2)

        db_distance = 0
        db_bearing = -1

        # distance
        if d_b_calc == "d" or d_b_calc == "d_b":
            db_a = (sin((db_lat2 - db_lat1) / 2) * sin((db_lat2 - db_lat1) / 2)) + (cos(db_lat1) * cos(db_lat2) *
                    sin((db_lon2 - db_lon1) / 2) * sin((db_lon2 - db_lon1) / 2))
            db_c = 2 * atan2(sqrt(db_a), sqrt(1 - db_a))
            db_distance = round(3956 * db_c)
        # bearing
        if d_b_calc == "b" or d_b_calc == "d_b":
            db_y = sin(db_lon2 - db_lon1) * cos(db_lat2)
            db_x = cos(db_lat1) * sin(db_lat2) - sin(db_lat1) * cos(db_lat2) * cos(db_lon2 - db_lon1)
            db_bearing = atan2(db_y, db_x)
            db_bearing = int(round((degrees(db_bearing) + 360) % 360))
        return db_distance, db_bearing
    except:
        return None, None


def bts_distance(acft_lat, bts_lat_dis, acft_lon, bts_lon_dis):
    try:
        acft_lat = radians(acft_lat)
        acft_lon = radians(acft_lon)
        bts_lat_dis = radians(bts_lat_dis)
        bts_lon_dis = radians(bts_lon_dis)

        bts_a = (sin((bts_lat_dis - acft_lat) / 2) * sin((bts_lat_dis - acft_lat) / 2)) + (cos(acft_lat) *
                                                                                           cos(bts_lat_dis) * sin(
                    (bts_lon_dis - acft_lon) / 2) * sin((bts_lon_dis - acft_lon) / 2))
        bts_c = 2 * atan2(sqrt(bts_a), sqrt(1 - bts_a))
        bts_d = round(3956 * bts_c)
        return bts_d
    except:
        return None


def angle_2_bts(heading, bearing, ant):
    if heading is not None or bearing is not None:
        if ant == "fwd":
            dif = (heading - 360) if heading > 180 else heading
            target = (bearing - dif) % 360
        else:
            heading = (heading - 180) % 360
            dif = (heading - 360) if heading > 180 else heading
            target = (bearing - dif) % 360
        return target
    else:
        return None

def location_bearing(lb_lat1, lb_lon1, lb_d, lb_bearing, R=6371):
    try:
        lb_d = lb_d * 1.609
        lb_lat1 = radians(lb_lat1)
        lb_lon1 = radians(lb_lon1)
        lb_a = radians(lb_bearing)
        lb_lat2 = asin(sin(lb_lat1) * cos(lb_d / R) + cos(lb_lat1) * sin(lb_d / R) * cos(lb_a))
        lb_lon2 = lb_lon1 + atan2(
            sin(lb_a) * sin(lb_d / R) * cos(lb_lat1),
            cos(lb_d / R) - sin(lb_lat1) * sin(lb_lat2)
        )
        return degrees(lb_lat2), degrees(lb_lon2)
    except:
        return None, None


def bts_re(bts_raw):
    try:
        pre = re.search(r'\d+', bts_raw)
        bts = pre.group(0) if pre else None
        post = re.search(r'-(\d+)$', bts_raw)
        sector = post.group(0) if post else None
        if (bts is not None and sector is not None) and 1 <= int(bts) < 500:
            return bts
        else:
            return False
    except:
        return False


def bts_info(bts):
    try:
        bts_raw = str(bts)
        bts = bts_re(bts_raw)
        if bts is not False:
            return bts
        else:
            return None
    except:
        return None


def bts_lat(bts_site):
    global df_cellsites
    try:
        if len(df_cellsites.query(f'Site_ID == {bts_site}')) >= 1:  # cell site found in cell database
            return df_cellsites.query(f'Site_ID == {bts_site}').iloc[0, 4]
        else:
            return None
    except:
        return None


def bts_lon(bts_site):
    global df_cellsites
    try:
        if len(df_cellsites.query(f'Site_ID == {bts_site}')) >= 1:  # cell site found in cell database
            return df_cellsites.query(f'Site_ID == {bts_site}').iloc[0, 3]
        else:
            return None
    except:
        return None


#creates a list of bts sites used throughout the flight
def bts_list(cell_list):  #creates single "list" AC1 & AC2 BTS connected to during flight
    bts_sites = []
    for site in cell_list:
        if site not in bts_sites:
            bts_sites.append(site)
    return bts_sites


#parses the file uploaded and verifies it is usable. Either populates global df_static (dataframe) or sets it to False
def parse_contents(contents, filename):
    global df_static
    global df_cellsites
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df_static = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        df_static.rename(columns={"lon 1": 'lon', "lat 1": "lat", "alt_m 1": "alt_m"}, inplace=True, errors='ignore')
        # format dataframe for our application
        df_static.drop(['RTT_Ping_Aircard2', 'acpu_time', 'act_set_pilot_eg', 'act_set_pilot_eg_aircard2',
                        'act_set_pilotpn_aircard2', 'agl_ft_1', 'agl_ft2', 'best_asp_sinr_buffer', 'time.1',
                        'best_asp_sinr_buffer_aircard2', 'npr_set_pilotpn', 'rpc', 'rpc_aircard2', 'agl_ft 2',
                        'cand_set_pilot_eg', 'cand_set_pilot_eg_aircard2', 'cand_set_pilotpn',
                        'cand_set_pilotpn_aircard2',
                        'coverage', 'drc_buffer', 'drc_buffer_aircard2', 'flight_state', 'flight_state_change',
                        'gps_admin_state', 'lon 2', 'lat 2', 'flight_number', 'gps_health', 'nbr_set_pilotpn',
                        'gps_health', 'gps_time', 'h_acpu_time', 'horizontal_velocity', 'hstr', 'hstr_aircard2',
                        'minute',
                        'nbr_set_pilot_eg', 'nbr_set_pilot_eg_aircard2', 'nbr_set_pilot_pn',
                        'nbr_set_pilot_pn_aircard2',
                        'new_coverage', 'pa_state', 'pa_state_aircard2', 'per_inst', 'per_inst_aircard2',
                        'per_sequence',
                        'per_sequence_aircard2', 'pilot_pn_asp', 'pilot_pn_asp_aircard2', 'pkt_rcvd_flag',
                        'pkt_rcvd_flag_aircard2', 'product_type', 'rpc_cell_index', 'rpc_cell_index_aircard2',
                        'serving_sector_id', 'vertical_velocity', 'write_time', 'new_time', 'type',
                        'serving_sector_pn', 'serving_sector_pn_aircard2', 'sw', 'act_set_pilotpn', 'new_coverage',
                        'agl_ft 1', 'act_set_pilotpn', 'alt_m 2', 'asp_filtered_sinr', 'asp_filtered_sinr_aircard2',
                        ], axis=1, inplace=True, errors='ignore')

        df_static.drop_duplicates(subset=['time'], keep='last', inplace=True)  # drop any duplicate times

        try:
            df_static['time'] = pd.to_datetime(df_static['time'], unit='ms')  # needed incase time is epoch format
        except:
            pass
        df_static['time'] = df_static['time'].astype('datetime64[ns]')
        df_static.sort_values(by='time', inplace=True)
        df_static.set_index('time', inplace=True)
        df_static = df_static.resample('min').asfreq(fill_value=None).reset_index()  # adds any missing lines by minute

        # calculate heading
        # find all rows with valid latitude and longitude
        coordinates_known = []
        df_static['lat'] = df_static['lat'].astype('float')
        df_static['lon'] = df_static['lon'].astype('float')
        for i in range(len(df_static)):
            if pd.isna(df_static.loc[i, 'lat']) is False and pd.isna(df_static.loc[i, 'lon']) is False:
                coordinates_known.append(i)
            else:
                df_static.loc[i, 'valid_data'] = False

        df_static.loc[
            coordinates_known, ['valid_data']] = True  # set all rows with good lat and long column valid_data to True
        # based on rows with good latitude and longitude calculate heading

        for i in range(len(coordinates_known) - 1):
            lat1 = df_static.loc[coordinates_known[i], 'lat']
            lat2 = df_static.loc[coordinates_known[(i + 1)], 'lat']
            lon1 = df_static.loc[coordinates_known[i], 'lon']
            lon2 = df_static.loc[coordinates_known[(i + 1)], 'lon']
            df_static.loc[coordinates_known[i + 1], "heading"] = distance_bearing(lat1, lat2, lon1, lon2, "b")[1]

        df_static[['heading']] = df_static[['heading']].ffill()  # forward fill in case last entry is missing
        df_static[['heading']] = df_static[['heading']].bfill()  # back fill heading of unknown headings
        df_static['heading'] = round(df_static['heading'] + 360 % 360).astype(int)

        # for missing data (latitude and longitude) interpolate latitude and longitude
        nan_index = []
        for i in range(len(df_static)):
            if df_static.loc[i, 'valid_data'] is not True:
                nan_index.append(i)
        if len(nan_index) > 0:
            a = np.array(nan_index)  # create numpy array of nan_index
            missing_row_group = np.split(a, np.where(np.diff(a) != 1)[0] + 1)  # groups consecutive rows missing
            for group in missing_row_group:
                row_a_lat = df_static.loc[group[0] - 1, 'lat']  # last known lat
                row_a_lon = df_static.loc[group[0] - 1, 'lon']  # last known lon
                row_b_lat = df_static.loc[group[-1] + 1, 'lat']  # next known lat
                row_b_lon = df_static.loc[group[-1] + 1, 'lon']  # next known lon
                distance_bearing_missing = distance_bearing(row_a_lat, row_b_lat, row_a_lon, row_b_lon, 'd_b')
                distance = distance_bearing_missing[0]  # distance of missing data
                bearing = distance_bearing_missing[1]  # carried heading for missing data actually unknown so carried
                segment = distance / (len(group) + 1)  # miles to each point, m = number of traces to fill
                for i in range(len(group)):
                    lat_long = location_bearing(df_static.loc[group[i] - 1, 'lat'], df_static.loc[group[i] - 1, 'lon'],
                                                segment,
                                                bearing)
                    df_static.loc[group[i], 'lat'] = lat_long[0]
                    df_static.loc[group[i], 'lon'] = lat_long[1]
                    df_static.loc[group[i], 'heading'] = bearing

        # altitude from meters to feet
        df_static[['alt_m']] = df_static[['alt_m']].ffill()  # next 2 lines altitude forward fill then back
        df_static[['alt_m']] = df_static[['alt_m']].bfill()
        df_static['alt_m'] = df_static['alt_m'].astype(float)
        df_static = df_static.assign(
            altitude=lambda x: (x['alt_m'] * 3.28084))  # calculate MSL altitude feet from meters
        df_static['altitude'] = df_static['altitude'].astype(int)

        # list and find coordinates for bts connected during flight for each aircard
        df_static['cell_search_id'] = df_static['cell_search_id'].apply(bts_info)
        df_static['cell_search_id_aircard2'] = df_static['cell_search_id_aircard2'].apply(bts_info)
        df_cellsites = pd.read_csv('cell_sites.csv')
        df_static["bts1_latitude"] = df_static['cell_search_id'].apply(bts_lat)
        df_static["bts1_longitude"] = df_static['cell_search_id'].apply(bts_lon)
        df_static["bts2_latitude"] = df_static['cell_search_id_aircard2'].apply(bts_lat)
        df_static["bts2_longitude"] = df_static['cell_search_id_aircard2'].apply(bts_lon)
        del df_cellsites  # no longer needed

        # distance and bearing to each bts for each aircard

        df_static["acft_distance_ac1_bts"] = df_static.apply(lambda row: distance_bearing(row['lat'],
                                                                                          row['bts1_latitude'],
                                                                                          row['lon'],
                                                                                          row['bts1_longitude'],
                                                                                          "d")[0], axis=1)

        df_static.loc[df_static["acft_distance_ac1_bts"] > 150, "acft_distance_ac1_bts"] = None


        df_static["acft_bearing_ac1_bts"] = df_static.apply(lambda row: distance_bearing(row['lat'],
                                                                                         row['bts1_latitude'],
                                                                                         row['lon'],
                                                                                         row['bts1_longitude'],
                                                                                         "b")[1], axis=1)

        df_static["acft_distance_ac2_bts"] = df_static.apply(lambda row: distance_bearing(row['lat'],
                                                                                          row['bts2_latitude'],
                                                                                          row['lon'],
                                                                                          row['bts2_longitude'],
                                                                                          "d")[0], axis=1)

        df_static.loc[df_static["acft_distance_ac2_bts"] > 150, "acft_distance_ac2_bts"] = None

        df_static["acft_bearing_ac2_bts"] = df_static.apply(lambda row: distance_bearing(row['lat'],
                                                                                         row['bts2_latitude'],
                                                                                         row['lon'],
                                                                                         row['bts2_longitude'],
                                                                                         "b")[1], axis=1)

        # compass bearing to bts for each aircard & antenna
        df_static["fwd_ac1_angle_2_bts"] = df_static.apply(lambda row: angle_2_bts(row['heading'],
                                                                                   row['acft_bearing_ac1_bts'],
                                                                              "fwd"), axis=1)

        df_static["fwd_ac2_angle_2_bts"] = df_static.apply(lambda row: angle_2_bts(row['heading'],
                                                                                   row['acft_bearing_ac2_bts'],
                                                                                   "fwd"), axis=1)

        df_static["aft_ac1_angle_2_bts"] = df_static.apply(lambda row: angle_2_bts(row['heading'],
                                                                                   row['acft_bearing_ac1_bts'],
                                                                                   "aft"), axis=1)

        df_static["aft_ac2_angle_2_bts"] = df_static.apply(lambda row: angle_2_bts(row['heading'],
                                                                                   row['acft_bearing_ac2_bts'],
                                                                                   "aft"), axis=1)
        df_static.to_csv("file.csv")
        return True
    except:
        print('nope')
        df_static = False
        return False


#returns a formated template of the hover page for the flight map
def flight_hover(minute_plot):
    global df_static
    try:
        df = df_static.copy(deep=True)
        if minute_plot == "5 minute":
            df = df.iloc[::5, :]

        template = []
        for i in df.index:
            time = df.loc[i, 'time'].strftime('%H:%M')
            altitude = f"{df.loc[i, 'altitude']:,.0f}"
            ac1drc = f"{df.loc[i, 'drc_kbps']:,.0f}"
            ac2drc = f"{df.loc[i, 'drc_kbps_aircard2']:,.0f}"
            ac1sinr = f"{df.loc[i, 'best_asp_sinr_buffer_calc']:,.1f}"
            ac2sinr = f"{df.loc[i, 'best_asp_sinr_buffer_aircard2_calc']:,.1f}"
            fwdac1rx = f"{df.loc[i, 'rx_agc0']:,.1f}"
            fwdac2rx = f"{df.loc[i, 'rx_agc0_aircard2']:,.1f}"
            aftac1rx = f"{df.loc[i, 'rx_agc1']:,.1f}"
            aftac2rx = f"{df.loc[i, 'rx_agc1_aircard2']:,.1f}"
            ac1tx = f"{df.loc[i, 'tx_agc']:.1f}"
            ac2tx = f"{df.loc[i, 'tx_agc_aircard2']:.1f}"
            ac1bts = df.loc[i, 'cell_search_id']
            ac2bts = df.loc[i, 'cell_search_id_aircard2']
            line1 = f'<BR><b>Time</b>: {time}  <b>Altitude</b>: {altitude} ft agl</BR>'
            line2 = f'<BR>AC1 DRC: {ac1drc} Kbps   AC2 DRC: {ac2drc} Kbps</BR>'
            line3 = f'<BR>AC1 SINR: {ac1sinr}   AC2 SINR: {ac2sinr}</BR>'
            line4 = f'<BR>AC1 FWD RX: {fwdac1rx} dB   AC2 FWD RX: {fwdac2rx} dB</BR>'
            line5 = f'<BR>AC1 AFT RX: {aftac1rx} dB   AC2 AFT RX: {aftac2rx} dB</BR>'
            line6 = f'<BR>TX: {ac1tx} dB      TX: {ac2tx} dB</BR>'
            line7 = f'<BR>AC1 BTS: {ac1bts}   AC2 BTS: {ac2bts}</BR>'
            template.append(line1 + line2 + line3 + line4 + line5 + line6 + line7)
        del df
        return template
    except:

        return PreventUpdate


#formats the hover label for the antenna plots
def antenna_hover(row):  #providing df_static row number
    global df_static
    template = []
    try:
        time = df_static.loc[row, 'time'].strftime('%H:%M')
        altitude = f"{df_static.loc[row, 'altitude']:,.0f}"
        for i in range(4):
            match i:
                case 0:
                    if df_static.loc[row, 'drc_kbps'] is not None:
                        drc = f"{df_static.loc[row, 'drc_kbps']:,.0f}"
                    else:
                        drc = "NR"
                    if df_static.loc[row, 'rx_agc0'] is not None:
                        rx = f"{df_static.loc[row, 'rx_agc0']:,.1f}"
                    else:
                        rx = "NR"
                    if df_static.loc[row, 'acft_distance_ac1_bts'] is not None:
                        dis = f"{df_static.loc[row, 'acft_distance_ac1_bts']:.0f}"
                    else:
                        dis = "NR"
                    if df_static.loc[row, 'fwd_ac1_angle_2_bts'] is not None:
                        angle = f"{df_static.loc[row, 'fwd_ac1_angle_2_bts']:.0f}"
                    else:
                        angle = "NR"
                    if df_static.loc[row, 'best_asp_sinr_buffer_calc'] is not None:
                        sinr = f"{df_static.loc[row, 'best_asp_sinr_buffer_calc']:,.1f}"
                    else:
                        sinr = "NR"
                    if df_static.loc[row, 'cell_search_id'] is not None:
                        bts = f"{df_static.loc[row, 'cell_search_id']}"
                    else:
                        bts = 'NR'
                    if df_static.loc[row, 'tx_agc'] is not None:
                        tx = f"{df_static.loc[row, 'tx_agc']:.1f}"
                    else:
                        tx = "NR"
                    line1 = f'<BR><b>Time</b>: {time}  <b>Altitude</b>: {altitude} ft agl</BR>'
                    line2 = f'<BR><b>DRC</b>: {drc} Kbps'
                    line3 = f'<BR><b>SINR</b>: {sinr}'
                    line4 = f'<BR><b>RX</b>: {rx} dB'
                    line5 = f'<BR><b>TX</b>: {tx} dB'
                    line6 = f'<BR><b>AC1 BTS</b>: {bts}'
                    line7 = f'<BR><b>Distance</b>: {dis} nm'
                    line8 = f'<BR><b>Bearing to BTS</b>: {angle}'
                    template.append(line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8)

                case 1:
                    if df_static.loc[row, 'drc_kbps_aircard2'] is not None:
                        drc = f"{df_static.loc[row, 'drc_kbps_aircard2']:,.0f}"
                    else:
                        drc = 'NR'
                    if df_static.loc[row, 'rx_agc0_aircard2'] is not None:
                        rx = f"{df_static.loc[row, 'rx_agc0_aircard2']:,.1f}"
                    else:
                        rx = "NR"
                    if df_static.loc[row, 'acft_distance_ac2_bts'] is not None:
                        dis = f"{df_static.loc[row, 'acft_distance_ac2_bts']:.0f}"
                    else:
                        dis = "NR"
                    if df_static.loc[row, 'fwd_ac2_angle_2_bts'] is not None:
                        angle = f"{df_static.loc[row, 'fwd_ac2_angle_2_bts']:.0f}"
                    else:
                        angle = 'NR'
                    if df_static.loc[row, 'best_asp_sinr_buffer_aircard2_calc'] is not None:
                        sinr = f"{df_static.loc[row, 'best_asp_sinr_buffer_aircard2_calc']:,.1f}"
                    else:
                        sinr = 'NR'
                    if df_static.loc[row, 'cell_search_id_aircard2'] is not None:
                        bts = f"{df_static.loc[row, 'cell_search_id_aircard2']}"
                    else:
                        bts = "NR"
                    if df_static.loc[row, 'tx_agc_aircard2'] is not None:
                        tx = f"{df_static.loc[row, 'tx_agc_aircard2']:.1f}"
                    else:
                        tx = 'NR'
                    line1 = f'<BR><b>Time</b>: {time}  <b>Altitude</b>: {altitude} ft agl</BR>'
                    line2 = f'<BR><b>DRC</b>: {drc} Kbps'
                    line3 = f'<BR><b>SINR</b>: {sinr}'
                    line4 = f'<BR><b>RX</b>: {rx} dB'
                    line5 = f'<BR><b>TX</b>: {tx} dB'
                    line6 = f'<BR><b>AC2 BTS</b>: {bts}'
                    line7 = f'<BR><b>Distance</b>: {dis} nm'
                    line8 = f'<BR><b>Bearing to BTS</b>: {angle}'
                    template.append(line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8)

                case 2:
                    if df_static.loc[row, 'drc_kbps'] is not None:
                        drc = f"{df_static.loc[row, 'drc_kbps']:,.0f}"
                    else:
                        drc = "NR"
                    if df_static.loc[row, 'rx_agc1'] is not None:
                        rx = f"{df_static.loc[row, 'rx_agc1']:,.1f}"
                    else:
                        rx = 'NR'
                    if df_static.loc[row, 'acft_distance_ac1_bts'] is not None:
                        dis = f"{df_static.loc[row, 'acft_distance_ac1_bts']:.0f}"
                    else:
                        dis = 'NR'
                    if df_static.loc[row, 'aft_ac1_angle_2_bts'] is not None:
                        angle = f"{df_static.loc[row, 'aft_ac1_angle_2_bts']:.0f}"
                    else:
                        angle = 'NR'
                    if df_static.loc[row, 'best_asp_sinr_buffer_calc'] is not None:
                        sinr = f"{df_static.loc[row, 'best_asp_sinr_buffer_calc']:,.1f}"
                    else:
                        sinr = "NR"
                    if df_static.loc[row, 'cell_search_id'] is not None:
                        bts = f"{df_static.loc[row, 'cell_search_id']}"
                    else:
                        bts = 'NR'
                    if df_static.loc[row, 'tx_agc'] is not None:
                        tx = f"{df_static.loc[row, 'tx_agc']:.1f}"
                    else:
                        tx = "NR"
                    line1 = f'<BR><b>Time</b>: {time}  <b>Altitude</b>: {altitude} ft agl</BR>'
                    line2 = f'<BR><b>DRC</b>: {drc} Kbps'
                    line3 = f'<BR><b>SINR</b>: {sinr}'
                    line4 = f'<BR><b>RX</b>: {rx} dB'
                    line5 = f'<BR><b>TX</b>: {tx} dB'
                    line6 = f'<BR><b>AC1 BTS</b>: {bts}'
                    line7 = f'<BR><b>Distance</b>: {dis} nm'
                    line8 = f'<BR><b>Bearing to BTS</b>: {angle}'
                    template.append(line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8)

                case 3:
                    if df_static.loc[row, 'drc_kbps_aircard2'] is not None:
                        drc = f"{df_static.loc[row, 'drc_kbps_aircard2']:,.0f}"
                    else:
                        drc = "NR"
                    if df_static.loc[row, 'rx_agc1_aircard2'] is not None:
                        rx = f"{df_static.loc[row, 'rx_agc1_aircard2']:,.1f}"
                    else:
                        rx = "NR"
                    if df_static.loc[row, 'acft_distance_ac2_bts'] is not None:
                        dis = f"{df_static.loc[row, 'acft_distance_ac2_bts']:.0f}"
                    else:
                        dis = "NR"
                    if df_static.loc[row, 'aft_ac2_angle_2_bts'] is not None:
                        angle = f"{df_static.loc[row, 'aft_ac2_angle_2_bts']:.0f}"
                    else:
                        angle = 'NR'
                    if df_static.loc[row, 'best_asp_sinr_buffer_aircard2_calc'] is not None:
                        sinr = f"{df_static.loc[row, 'best_asp_sinr_buffer_aircard2_calc']:,.1f}"
                    else:
                        sinr = "NR"
                    if df_static.loc[row, 'cell_search_id_aircard2'] is not None:
                        bts = f"{df_static.loc[row, 'cell_search_id_aircard2']}"
                    else:
                        bts = 'NR'
                    if df_static.loc[row, 'tx_agc_aircard2'] is not None:
                        tx = f"{df_static.loc[row, 'tx_agc_aircard2']:.1f}"
                    else:
                        tx = 'NR'
                    line1 = f'<BR><b>Time</b>: {time}  <b>Altitude</b>: {altitude} ft agl</BR>'
                    line2 = f'<BR><b>DRC</b>: {drc} Kbps'
                    line3 = f'<BR><b>SINR</b>: {sinr}'
                    line4 = f'<BR><b>RX</b>: {rx} dB'
                    line5 = f'<BR><b>TX</b>: {tx} dB'
                    line6 = f'<BR><b>AC2 BTS</b>: {bts}'
                    line7 = f'<BR><b>Distance</b>: {dis} nm'
                    line8 = f'<BR><b>Bearing to BTS</b>: {angle}'
                    template.append(line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8)
        return template
    except:
        return template


#colors the flight map plots
def color_plot(minute_plot, trace):
    global df_static
    try:
        df = df_static.copy(deep=True)
        if minute_plot == "5 minute":
            df = df.iloc[::5, :]
        color = []
        for i in df.index:
            match trace:
                case 'Flight':
                    if df.loc[i, 'valid_data'] == True:
                        color.append('green')
                    else:
                        color.append('black')
                case 'DRC':
                    ac1_kbps = pd.to_numeric(df.loc[i, 'drc_kbps'], errors='coerce')
                    ac2_kbps = pd.to_numeric(df.loc[i, 'drc_kbps_aircard2'], errors='coerce')
                    if df.loc[i, 'valid_data'] == False:
                        color.append('black')
                    elif ac1_kbps >= 1700 or ac2_kbps >= 1700:
                        color.append('green')
                    elif ac1_kbps >= 1200 or ac2_kbps >= 1200:
                        color.append('gold')
                    else:
                        color.append('red')
                case 'SINR':
                    ac1_sinr = pd.to_numeric(df.loc[i, 'best_asp_sinr_buffer_calc'], errors='coerce')
                    ac2_sinr = pd.to_numeric(df.loc[i, 'best_asp_sinr_buffer_aircard2_calc'], errors='coerce')
                    if df.loc[i, 'valid_data'] == False:
                        color.append('black')
                    elif ac1_sinr >= 10 or ac2_sinr >= 10:
                        color.append('green')
                    elif ac1_sinr >= 5 or ac2_sinr >= 5:
                        color.append('gold')
                    elif ac1_sinr >= 1 or ac2_sinr >= 1:
                        color.append('orange')
                    else:
                        color.append('red')
                case 'RX':
                    fwd_ac1_rx = pd.to_numeric(df.loc[i, 'rx_agc0'], errors='coerce')
                    fwd_ac2_rx = pd.to_numeric(df.loc[i, 'rx_agc0_aircard2'], errors='coerce')
                    aft_ac1_rx = pd.to_numeric(df.loc[i, 'rx_agc1'], errors='coerce')
                    aft_ac2_rx = pd.to_numeric(df.loc[i, 'rx_agc1_aircard2'], errors='coerce')
                    if df.loc[i, 'valid_data'] == False:
                        color.append('black')
                    elif (fwd_ac1_rx >= -70 or aft_ac1_rx >= -70) and (fwd_ac2_rx >= -70 or aft_ac2_rx >= -70):
                        color.append('green')
                    elif (fwd_ac1_rx >= -80 or aft_ac1_rx >= -80) and (fwd_ac2_rx >= -80 or aft_ac2_rx >= -80):
                        color.append('gold')
                    elif (fwd_ac1_rx >= -90 or aft_ac1_rx >= -90) and (fwd_ac2_rx >= -90 or aft_ac2_rx >= -90):
                        color.append('orange')
                    else:
                        color.append('red')
                case 'TX':
                    ac1_tx = pd.to_numeric(df.loc[i, 'tx_agc'], errors='coerce')
                    ac2_tx = pd.to_numeric(df.loc[i, 'tx_agc_aircard2'], errors='coerce')
                    if df.loc[i, 'valid_data'] == False:
                        color.append('black')
                    elif ac1_tx >= 1 or ac2_tx >= 1:
                        color.append('green')
                    elif ac1_tx >= -5 or ac2_tx >= -5:
                        color.append('gold')
                    elif ac1_tx >= -12 or ac2_tx >= -12:
                        color.append('orange')
                    else:
                        color.append('red')
        del df
        return color
    except:
        return PreventUpdate


#colors the antenna plots
def color_airborne(row, trace):
    global df_static
    color = ['', '', '', '']
    try:
        match trace:
            case 'Flight':
                for i in range(4):
                    color[i] = 'blue'
            case 'DRC':
                ac1_kbps = pd.to_numeric(df_static.loc[row, 'drc_kbps'], errors='coerce')
                ac2_kbps = pd.to_numeric(df_static.loc[row, 'drc_kbps_aircard2'], errors='coerce')
                if math.isnan(ac1_kbps) is True:
                    color[0] = 'black'
                    color[2] = 'black'
                elif ac1_kbps >= 1700:
                    color[0] = 'green'
                    color[2] = 'green'
                elif ac1_kbps >= 1300:
                    color[0] = 'gold'
                    color[2] = 'gold'
                else:
                    color[0] = 'red'
                    color[2] = 'red'
                if math.isnan(ac2_kbps) is True:
                    color[1] = 'black'
                    color[3] = 'black'
                elif ac2_kbps >= 1700:
                    color[1] = 'green'
                    color[3] = 'green'
                elif ac2_kbps >= 1300:
                    color[1] = 'gold'
                    color[3] = 'gold'
                else:
                    color[1] = 'red'
                    color[3] = 'red'
            case 'SINR':
                ac1_sinr = pd.to_numeric(df_static.loc[row, 'best_asp_sinr_buffer_calc'], errors='coerce')
                ac2_sinr = pd.to_numeric(df_static.loc[row, 'best_asp_sinr_buffer_aircard2_calc'], errors='coerce')
                if math.isnan(ac1_sinr) is True:
                    color[0] = 'black'
                    color[2] = 'black'
                elif ac1_sinr >= 10:
                    color[0] = 'green'
                    color[2] = 'green'
                elif ac1_sinr >= 5:
                    color[0] = 'gold'
                    color[2] = 'gold'
                elif ac1_sinr >= 1:
                    color[0] = 'orange'
                    color[2] = 'orange'
                else:
                    color[0] = 'red'
                    color[2] = 'red'
                if math.isnan(ac2_sinr) is True:
                    color[1] = 'black'
                    color[3] = 'black'
                elif ac2_sinr >= 10:
                    color[1] = 'green'
                    color[3] = 'green'
                elif ac2_sinr >= 5:
                    color[1] = 'gold'
                    color[3] = 'gold'
                elif ac2_sinr >= 1:
                    color[1] = 'orange'
                    color[3] = 'orange'
                else:
                    color[1] = 'red'
                    color[3] = 'red'
            case 'RX':
                fwd_ac1_rx = pd.to_numeric(df_static.loc[row, 'rx_agc0'], errors='coerce')
                fwd_ac2_rx = pd.to_numeric(df_static.loc[row, 'rx_agc0_aircard2'], errors='coerce')
                aft_ac1_rx = pd.to_numeric(df_static.loc[row, 'rx_agc1'], errors='coerce')
                aft_ac2_rx = pd.to_numeric(df_static.loc[row, 'rx_agc1_aircard2'], errors='coerce')
                if math.isnan(fwd_ac1_rx) is True:
                    color[0] = 'black'
                elif fwd_ac1_rx >= -70:
                    color[0] = 'green'
                elif fwd_ac1_rx >= -80:
                    color[0] = 'gold'
                elif fwd_ac1_rx >= -90:
                    color[0] = 'orange'
                else:
                    color[0] = 'red'
                if math.isnan(fwd_ac2_rx) is True:
                    color[1] = 'black'
                elif fwd_ac2_rx >= -70:
                    color[1] = 'green'
                elif fwd_ac2_rx >= -80:
                    color[1] = 'gold'
                elif fwd_ac2_rx >= -90:
                    color[1] = 'orange'
                else:
                    color[1] = 'red'
                if math.isnan(aft_ac1_rx) is True:
                    color[2] = 'black'
                elif aft_ac1_rx >= -70:
                    color[2] = 'green'
                elif aft_ac1_rx >= -80:
                    color[2] = 'gold'
                elif aft_ac1_rx >= -90:
                    color[2] = 'orange'
                else:
                    color[2] = 'red'
                if math.isnan(aft_ac2_rx) is True:
                    color[3] = 'black'
                elif aft_ac2_rx >= -70:
                    color[3] = 'green'
                elif aft_ac2_rx >= -80:
                    color[3] = 'gold'
                elif aft_ac2_rx >= -90:
                    color[3] = 'orange'
                else:
                    color[3] = 'red'
            case 'TX':
                ac1_tx = pd.to_numeric(df_static.loc[row, 'tx_agc'], errors='coerce')
                ac2_tx = pd.to_numeric(df_static.loc[row, 'tx_agc_aircard2'], errors='coerce')
                if math.isnan(ac1_tx) is True:
                    color[0] = 'black'
                    color[2] = 'black'
                elif ac1_tx >= 1:
                    color[0] = 'green'
                    color[2] = 'green'
                elif ac1_tx >= -5:
                    color[0] = 'gold'
                    color[2] = 'gold'
                elif ac1_tx >= -12:
                    color[0] = 'orange'
                    color[2] = 'orange'
                else:
                    color[0] = 'red'
                    color[2] = 'red'
                if math.isnan(ac2_tx) is True:
                    color[1] = 'black'
                    color[3] = 'black'
                elif ac2_tx >= 1:
                    color[1] = 'green'
                    color[3] = 'green'
                elif ac2_tx >= -5:
                    color[1] = 'gold'
                    color[3] = 'gold'
                elif ac2_tx >= -12:
                    color[1] = 'orange'
                    color[3] = 'orange'
                else:
                    color[1] = 'red'
                    color[3] = 'red'
        return color
    except:
        return color


#plots the maps and or antenna plot when there is data
def plotter(minute_plot, time_start, time_end, plot_choice, trace, rx_plot, bts_plot):
    global df_static
    #time_start and time_end are df_static row indexes
    try:
        df = df_static.copy(deep=True)
        df = df.iloc[time_start:time_end]
        df = df.reset_index()
        if minute_plot == "5 minute":
            df = df.iloc[::5, :]
            df = df.reset_index()
        # create a dataframe for BTS sites connected with during flight single list for both AC1 and AC2
        cell1_list = df['cell_search_id'].values.tolist()
        cell2_list = df['cell_search_id_aircard2'].values.tolist()

        cell_list = bts_list((cell1_list + cell2_list))
        cell_list = [item for item in cell_list if not (pd.isnull(item)) == True]  # remove any NaN from lists
        cell_list = [x for x in cell_list if x != 0]  # removes any zeros
        cell_list.sort(key=int)
        df_btslist = pd.DataFrame(cell_list, columns=['Site_ID'])
        df_cellsites = pd.read_csv('cell_sites.csv')
        df_btslist['Site_Name'] = None
        df_btslist['latitude'] = None
        df_btslist['longitude'] = None
        for i in range(len(df_btslist)):
            bts = str(df_btslist.loc[i, 'Site_ID'])
            if len(df_cellsites.query(f'Site_ID == {bts}')) >= 1:
                df_btslist.loc[i, 'Site_Name'] = df_cellsites.query(f'Site_ID == {bts}').iloc[0, 0]
                df_btslist.loc[i, 'latitude'] = df_cellsites.query(f'Site_ID == {bts}').iloc[0, 4]
                df_btslist.loc[i, 'longitude'] = df_cellsites.query(f'Site_ID == {bts}').iloc[0, 3]
            else:
                df_btslist.loc[i, 'Site_Name'] = None
                df_btslist.loc[i, 'latitude'] = None
                df_btslist.loc[i, 'longitude'] = None

        if plot_choice == 'Map':  #flight plt
            # df.reset_index(drop=True, inplace=True)
            flight_hover_template = flight_hover(minute_plot)
            trace_color = color_plot(minute_plot, trace)

            fig = go.Figure(data=go.Scattergeo(
                lon=df['lon'],
                lat=df['lat'],
                marker=dict(size=15, symbol='arrow-up', angle=df['heading'], color=trace_color),
                showlegend=False,
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=10,
                    font_family="Rockwell",
                ),
                hovertemplate=flight_hover_template,
                name=''))
            # add BTS to flight map
            if bts_plot is True:
                fig.add_scattergeo(
                    lon=df_btslist['longitude'],
                    lat=df_btslist['latitude'],
                    marker=dict(size=35, symbol='circle', color='#2aa198', opacity=0.75, line=dict(width=0)),
                    name="",
                    showlegend=False,
                    customdata=np.stack((df_btslist['Site_Name'], df_btslist['Site_ID']), axis=-1),
                    hovertemplate='<b>Site Name</b>: %{customdata[0]}<br>' +
                                  '<b>Site ID</b>: %{customdata[1]}<br>'
                )
            #plot aircard traces to BTS, if both same green
            if bts_plot is True:
                for i in range(len(df)):
                    if ((df.loc[i, 'cell_search_id']) is not None and
                            df.loc[i, 'cell_search_id'] == df.loc[i, 'cell_search_id_aircard2']):
                        fig.add_scattergeo(
                            lon=[df.loc[i, 'lon'], df.loc[i, 'bts1_longitude']],
                            lat=[df.loc[i, 'lat'], df.loc[i, 'bts1_latitude']],
                            mode='lines',
                            line=dict(width=1, color='#2aa198'),
                            opacity=1,
                            name='',
                            showlegend=False,
                            hoverinfo='none'
                        )
                    else:
                        if df.loc[i, 'cell_search_id'] is not None:
                            fig.add_scattergeo(
                                lon=[df.loc[i, 'lon'], df.loc[i, 'bts1_longitude']],
                                lat=[df.loc[i, 'lat'], df.loc[i, 'bts1_latitude']],
                                mode='lines',
                                line=dict(width=1, color='#268bd2'),
                                opacity=1,
                                name='',
                                showlegend=False,
                                hoverinfo='none'
                            )
                        if df.loc[i, 'cell_search_id_aircard2'] is not None:
                            fig.add_scattergeo(
                                lon=[df.loc[i, 'lon'], df.loc[i, 'bts2_longitude']],
                                lat=[df.loc[i, 'lat'], df.loc[i, 'bts2_latitude']],
                                mode='lines',
                                line=dict(width=1, color='#b58900'),
                                opacity=1,
                                name='',
                                showlegend=False,
                                hoverinfo='none'
                            )
            fig.update_layout(
                geo=dict(
                    resolution=50,
                    scope='north america',
                    showcountries=True,
                    showsubunits=True, subunitcolor="#839496",
                    bgcolor='#839496'),
                hoverlabel=dict(
                    bgcolor='#63666A',
                    font_size=16,
                    font_family='Rockwell',
                ),
                autosize=True,
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(yanchor="middle", y=.5),
                paper_bgcolor='#002b36'
            )

            fig.update_geos(fitbounds="locations")
            fig.update_yaxes(automargin=True)
            del df_cellsites
            del df
            return fig

        else:  #antenna plot
            fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'polar'}] * 2] * 2, subplot_titles=('FWD Aircard 1',
                                                                                                     'FWD Aircard 2',
                                                                                                     'AFT Aircard 1',
                                                                                                     'AFT Aircard 2'))

            trace_marker = dict(symbol='arrow', size=15, angleref='previous')
            trace_hoverlabel = dict(bgcolor="white", font_size=10, font_family="Rockwell")
            trace_mode = 'lines+markers'
            for i in range(len(df)):
                df_static_row = df_static.index[df_static['time'] == df.loc[i, "time"]].to_list()
                template = antenna_hover(df_static_row[0])
                line_color = color_airborne(df_static_row[0], trace)

                fwd_ac1_rx = df.loc[i, 'rx_agc0']
                fwd_ac2_rx = df.loc[i, 'rx_agc0_aircard2']
                aft_ac1_rx = df.loc[i, 'rx_agc1']
                aft_ac2_rx = df.loc[i, 'rx_agc1_aircard2']
                for x in range(2):
                    for y in range(2):
                        if x == 0 and y == 0:
                            if rx_plot is True and fwd_ac1_rx < aft_ac1_rx:
                                break
                            elif (df.loc[i, 'acft_distance_ac1_bts'] is not None and
                                  df.loc[i, "fwd_ac1_angle_2_bts"] is not None):
                                fig.add_trace((go.Scatterpolar(
                                    r=[0, df.loc[i, 'acft_distance_ac1_bts']],
                                    theta=[0, df.loc[i, "fwd_ac1_angle_2_bts"]],
                                    mode=trace_mode,
                                    line=dict(width=2, color=line_color[0]),
                                    marker=trace_marker,
                                    name='',
                                    showlegend=False,
                                    hoverlabel=trace_hoverlabel,
                                    hovertemplate=template[0],
                                )), row=x + 1, col=y + 1)
                                fig.layout.annotations[0].update(x=.1)
                                fig.update_polars(
                                    radialaxis_tickfont_size=10,
                                    angularaxis=dict(
                                        tickfont_size=12,
                                        rotation=90,
                                        direction="clockwise"
                                    ), row=x + 1, col=y + 1)
                        if x == 0 and y == 1:
                            if rx_plot is True and fwd_ac2_rx < aft_ac2_rx:
                                break
                            elif (df.loc[i, 'acft_distance_ac2_bts'] is not None and
                                  df.loc[i, "fwd_ac2_angle_2_bts"] is not None):
                                fig.add_trace((go.Scatterpolar(
                                    r=[0, df.loc[i, 'acft_distance_ac2_bts']],
                                    theta=[0, df.loc[i, "fwd_ac2_angle_2_bts"]],
                                    mode=trace_mode,
                                    line=dict(width=2, color=line_color[1]),
                                    marker=trace_marker,
                                    name='',
                                    showlegend=False,
                                    hoverlabel=trace_hoverlabel,
                                    hovertemplate=template[1],
                                )), row=x + 1, col=y + 1)
                                fig.layout.annotations[1].update(x=.6)
                                fig.update_polars(
                                    radialaxis_tickfont_size=10,
                                    angularaxis=dict(
                                        tickfont_size=12,
                                        rotation=90,
                                        direction="clockwise"
                                    ), row=x + 1, col=y + 1)
                        if x == 1 and y == 0:
                            if rx_plot is True and aft_ac1_rx < fwd_ac1_rx:
                                break
                            elif (df.loc[i, 'acft_distance_ac1_bts'] is not None and
                                  df.loc[i, "aft_ac1_angle_2_bts"] is not None):
                                fig.add_trace((go.Scatterpolar(
                                    r=[0, df.loc[i, 'acft_distance_ac1_bts']],
                                    theta=[0, df.loc[i, "aft_ac1_angle_2_bts"]],
                                    mode=trace_mode,
                                    line=dict(width=2, color=line_color[2]),
                                    marker=trace_marker,
                                    name='',
                                    showlegend=False,
                                    hoverlabel=trace_hoverlabel,
                                    hovertemplate=template[2],
                                )), row=x + 1, col=y + 1)
                                fig.layout.annotations[2].update(x=.1)
                                fig.update_polars(
                                    radialaxis_tickfont_size=10,
                                    angularaxis=dict(
                                        tickfont_size=12,
                                        rotation=270,
                                        direction="clockwise"
                                    ), row=x + 1, col=y + 1)
                        if x == 1 and y == 1:
                            if rx_plot is True and aft_ac2_rx < fwd_ac2_rx:
                                break
                            elif (df.loc[i, 'acft_distance_ac2_bts'] is not None and
                                  df.loc[i, "aft_ac2_angle_2_bts"] is not None):
                                fig.add_trace((go.Scatterpolar(
                                    r=[0, df.loc[i, 'acft_distance_ac2_bts']],
                                    theta=[0, df.loc[i, "aft_ac2_angle_2_bts"]],
                                    mode=trace_mode,
                                    line=dict(width=2, color=line_color[3]),
                                    marker=trace_marker,
                                    name='',
                                    showlegend=False,
                                    hoverlabel=trace_hoverlabel,
                                    hovertemplate=template[3],
                                )), row=x + 1, col=y + 1)
                                fig.layout.annotations[3].update(x=.6)
                                fig.update_polars(
                                    radialaxis_tickfont_size=10,
                                    angularaxis=dict(
                                        tickfont_size=12,
                                        rotation=270,
                                        direction="clockwise"
                                    ), row=x + 1, col=y + 1)

            fig.update_layout(
                title={
                    'text': 'Antenna Plot',
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'x': 0.2

                },
                showlegend=False,
                height=800,
                width=800,
                autosize=False)
            fig.update_yaxes(automargin=True)
            del df_cellsites
            del df
            return fig
    except:
        print("problem in plot")
        raise PreventUpdate


#configures the time slider
def time_slider_config(minute_plot):  # range slider, value integers as list should only setup slider
    global df_static
    marker = {}
    try:
        df = df_static.copy(deep=True)
        # if minute_plot == "5 minute":
        #     df = df.iloc[::5, :]

        time_start = df['time'].iloc[0]  #time start
        time_end = df['time'].iloc[-1]  #time end
        df.index = pd.RangeIndex(len(df.index))
        df.index = range(len(df.index))
        df_length = len(df)
        slider_options = dict((d_key, d_val) for d_key, d_val in enumerate(df['time'].unique()))
        match df_length:
            case df_length if df_length <= 60:
                scale = .5
            case df_length if df_length <= 120:
                scale = .25
            case df_length if df_length <= 180:
                scale = .1
            case df_length if df_length <= 240:
                scale = .05
            case df_length if df_length > 240:
                scale = .025
            case _:
                scale = .5

        x = np.linspace(min(slider_options.keys()), max(slider_options.keys()), num=int(len(df) * scale), dtype=int)

        for i in x:
            formated_date_time = df.loc[i, 'time'].strftime('%H:%M')
            marker[str(i)] = {"label": formated_date_time,
                              "style": {"transform": "rotate(45deg)", "white-space": "nowrap"}}

        if len(df) > 0:
            minimum = min(slider_options.keys())
            maximum = max(slider_options.keys())
        else:
            minimum = 0
            maximum = 0

        value = [min(slider_options.keys()), max(slider_options.keys())]
        del df
        return [time_start, time_end, marker, minimum, maximum, value]
    except:
        return [0, 0, marker, 0, 0, [0, 0]]


#formats the legend
def legend_plotter(plot_choice, trace):
    legend = []
    html_style = []
    match plot_choice:
        case 'Map':
            match trace:
                case 'Flight':
                    legend.append('Reported Data')
                    legend.append('Missing Data')
                    legend.append('')
                    legend.append('')
                    legend.append('')
                    html_style.append({'color': 'green'})
                    html_style.append({'color': 'black'})
                    html_style.append({})
                    html_style.append({})
                    html_style.append({})
                    return legend, html_style
                case 'DRC':
                    legend.append('AC1 or AC2 >= 1700 Kbps')
                    legend.append('AC1 or AC2 >= 1200 Kbps')
                    legend.append('AC1 and AC2 < 1200 Kbps')
                    legend.append('Missing Data')
                    legend.append('')
                    html_style.append({'color': 'green'})
                    html_style.append({'color': 'gold'})
                    html_style.append({'color': 'red'})
                    html_style.append({'color': 'black'})
                    html_style.append({})
                    return legend, html_style
                case 'SINR':
                    legend.append('AC1 or AC2 >= 10')
                    legend.append('AC1 or AC2 >= 5')
                    legend.append('AC1 and AC2 >= 1')
                    legend.append('AC1 and AC2 < 1')
                    legend.append('Missing Data')
                    html_style.append({'color': 'green'})
                    html_style.append({'color': 'gold'})
                    html_style.append({'color': 'orange'})
                    html_style.append({'color': 'red'})
                    html_style.append({'color': 'black'})
                    return legend, html_style
                case 'RX':
                    legend.append('AC1 or AC2 >= -70 dB')
                    legend.append('AC1 or AC2 >= -80 dB')
                    legend.append('AC1 and AC2 >= -90 dB')
                    legend.append('AC1 and AC2 < -90 dB')
                    legend.append('Missing Data')
                    html_style.append({'color': 'green'})
                    html_style.append({'color': 'gold'})
                    html_style.append({'color': 'orange'})
                    html_style.append({'color': 'red'})
                    html_style.append({'color': 'black'})
                    return legend, html_style
                case 'TX':
                    legend.append('AC1 or AC2 >= 1 dB')
                    legend.append('AC1 or AC2 >= -5 dB')
                    legend.append('AC1 and AC2 >= -12 dB')
                    legend.append('AC1 and AC2 < -12 dB')
                    legend.append('Missing Data')
                    html_style.append({'color': 'green'})
                    html_style.append({'color': 'gold'})
                    html_style.append({'color': 'orange'})
                    html_style.append({'color': 'red'})
                    html_style.append({'color': 'black'})
                    return legend, html_style
        case 'Antenna':
            match trace:
                case 'Flight':
                    legend.append('Reported Data')
                    legend.append('')
                    legend.append('')
                    legend.append('')
                    legend.append('')
                    html_style.append({'color': 'blue'})
                    html_style.append({})
                    html_style.append({})
                    html_style.append({})
                    html_style.append({})
                    return legend, html_style
                case 'DRC':
                    legend.append('AC DRC >= 1700 Kbps')
                    legend.append('AC DRC >= 1200 Kbps')
                    legend.append('AC DRC < 1200 Kbps')
                    legend.append('Missing Data')
                    legend.append('')
                    html_style.append({'color': 'green'})
                    html_style.append({'color': 'gold'})
                    html_style.append({'color': 'red'})
                    html_style.append({'color': 'black'})
                    html_style.append({})
                    return legend, html_style
                case 'SINR':
                    legend.append('AC SINR >= 10')
                    legend.append('AC SINR >= 5')
                    legend.append('AC SINR >= 1')
                    legend.append('AC SINR < 1')
                    legend.append('Missing Data')
                    html_style.append({'color': 'green'})
                    html_style.append({'color': 'gold'})
                    html_style.append({'color': 'orange'})
                    html_style.append({'color': 'red'})
                    html_style.append({'color': 'black'})
                    return legend, html_style
                case 'RX':
                    legend.append('ANT AC RX >= -70 dB')
                    legend.append('ANT AC RX >= -80 dB')
                    legend.append('ANT AC RX >= -90 dB')
                    legend.append('ANT AC RX < -90 dB')
                    legend.append('Missing Data')
                    html_style.append({'color': 'green'})
                    html_style.append({'color': 'gold'})
                    html_style.append({'color': 'orange'})
                    html_style.append({'color': 'red'})
                    html_style.append({'color': 'black'})
                    return legend, html_style
                case 'TX':
                    legend.append('AC TX >= 1 dB')
                    legend.append('AC TX >= -5 dB')
                    legend.append('AC TX >= -12 dB')
                    legend.append('AC TX < -12 dB')
                    legend.append('Missing Data')
                    html_style.append({'color': 'green'})
                    html_style.append({'color': 'gold'})
                    html_style.append({'color': 'orange'})
                    html_style.append({'color': 'red'})
                    html_style.append({'color': 'black'})
                    return legend, html_style
        case 'empty':
            match trace:
                case 'Flight':
                    legend.append('')
                    legend.append('')
                    legend.append('')
                    legend.append('')
                    legend.append('')
                    html_style.append({})
                    html_style.append({})
                    html_style.append({})
                    html_style.append({})
                    html_style.append({})
                    return legend, html_style


#grabs the first valid report of the LRU SN
def find_lru_sn():
    global df_static
    lru = None
    i = 0
    while lru is None and i <= (len(df_static) - 1):
        lru = str(df_static.loc[i, 'tail'])
        i = i + 1
    return lru


#fills the xpol and service minute table
def xpol_fill(minute_plot, time_start, time_end):
    global df_static
    try:
        df = df_static.copy(deep=True)
        df = df.iloc[time_start:time_end]
        df = df.reset_index()

        fwd_hpol_xpol_total = 0
        fwd_vpol_xpol_total = 0
        aft_hpol_xpol_total = 0
        aft_vpol_xpol_total = 0
        hpol_total_count = 0
        vpol_total_count = 0
        hpol_srv_min = 0
        vpol_srv_min = 0
        flight_minutes = 0

        if minute_plot == "5 minute":
            df = df.iloc[::5, :]
            df = df.reset_index()

        last_row_index = df.index[-1]

        # find flight minutes
        flight_minutes = int((df['time'].iloc[last_row_index] - df['time'].iloc[0]) /
                             pd.Timedelta(minutes=1))
        # find number of times drc is above 0 gives service minutes aircard was usable
        start_time = df.loc[0, 'time']
        end_time = df.loc[last_row_index, 'time']
        start_time_index = df_static.index[(df_static['time'] - start_time).abs().argsort()[:1]].tolist()
        end_time_index = df_static.index[(df_static['time'] - end_time).abs().argsort()[:1]].tolist()

        for i in range(start_time_index[0], end_time_index[0]):
            if df_static.loc[i, "drc_kbps"] > 0:
                hpol_srv_min = hpol_srv_min + 1
            if df_static.loc[i, "drc_kbps_aircard2"] > 0:
                vpol_srv_min = vpol_srv_min + 1

        hpol_total_drc = df_static['drc_kbps'].iloc[start_time_index[0]:end_time_index[0]].sum()
        hpol_avg_drc = hpol_total_drc / hpol_srv_min if hpol_srv_min else 0
        hpol_avg_drc = f"{hpol_avg_drc:.0f}"
        vpol_total_drc = df_static['drc_kbps_aircard2'].iloc[start_time_index[0]:end_time_index[0]].sum()
        vpol_avg_drc = vpol_total_drc / vpol_srv_min if vpol_srv_min else 0
        vpol_avg_drc = f"{vpol_avg_drc:.0f}"

        for i in range(start_time_index[0], end_time_index[0]):
            #sum Hpol DRC
            if df_static.loc[i, 'drc_kbps'] is not None and df_static.loc[i, 'drc_kbps'] > 0:
                hpol_total_count = hpol_total_count + 1
                if df_static.loc[i, 'rx_agc0'] > df_static.loc[i, 'rx_agc1']:
                    fwd_hpol_xpol_total = fwd_hpol_xpol_total + 1
                elif df_static.loc[i, 'rx_agc0'] < df_static.loc[i, 'rx_agc1']:
                    aft_hpol_xpol_total = aft_hpol_xpol_total + 1
                else:
                    hpol_total_count = hpol_total_count - 1
            #sum Vpol DRC
            if df_static.loc[i, 'drc_kbps_aircard2'] is not None and df_static.loc[i, 'drc_kbps_aircard2'] > 0:
                vpol_total_count = vpol_total_count + 1
                if df_static.loc[i, 'rx_agc0_aircard2'] > df_static.loc[i, 'rx_agc1_aircard2']:
                    fwd_vpol_xpol_total = fwd_vpol_xpol_total + 1
                elif df_static.loc[i, 'rx_agc0_aircard2'] < df_static.loc[i, 'rx_agc1_aircard2']:
                    aft_vpol_xpol_total = aft_vpol_xpol_total + 1
                else:
                    vpol_total_count = vpol_total_count - 1

        fwd_hpol_xpol = fwd_hpol_xpol_total / hpol_total_count if hpol_total_count else 0
        aft_hpol_xpol = aft_hpol_xpol_total / hpol_total_count if hpol_total_count else 0
        fwd_vpol_xpol = fwd_vpol_xpol_total / vpol_total_count if vpol_total_count else 0
        aft_vpol_xpol = aft_vpol_xpol_total / vpol_total_count if vpol_total_count else 0

        fwd_hpol_xpol = f"{(fwd_hpol_xpol * 100):.0f}%" if fwd_hpol_xpol else None
        fwd_vpol_xpol = f"{(fwd_vpol_xpol * 100):.0f}%" if fwd_vpol_xpol else None
        aft_hpol_xpol = f"{(aft_hpol_xpol * 100):.0f}%" if aft_hpol_xpol else None
        aft_vpol_xpol = f"{(aft_vpol_xpol * 100):.0f}%" if aft_vpol_xpol else None

        if fwd_hpol_xpol is None:
            fwd_hpol_xpol = "0%"
        if fwd_vpol_xpol is None:
            fwd_vpol_xpol = "0%"
        if aft_hpol_xpol is None:
            aft_hpol_xpol = "0%"
        if aft_vpol_xpol is None:
            aft_vpol_xpol = "0%"

        hpol_srv_min = f"{hpol_srv_min:.0f}" if hpol_srv_min else None
        vpol_srv_min = f"{vpol_srv_min:.0f}" if vpol_srv_min else None

        if hpol_srv_min is None:
            hpol_srv_min = "0"
        if vpol_srv_min is None:
            vpol_srv_min = "0"

        xpol_data = []
        xpol_data.append(hpol_srv_min)
        xpol_data.append(hpol_avg_drc)
        xpol_data.append(vpol_srv_min)
        xpol_data.append(vpol_avg_drc)
        xpol_data.append(fwd_hpol_xpol)
        xpol_data.append(fwd_vpol_xpol)
        xpol_data.append(aft_hpol_xpol)
        xpol_data.append(aft_vpol_xpol)
        xpol_data.append(flight_minutes)

        del df
        return xpol_data
    except:
        xpol_data = []
        for i in range(9):
            xpol_data.append("")
            return xpol_data


server = Flask(__name__)

app = dash.Dash(name='app1', server=server, external_stylesheets=[dbc.themes.SOLAR])

# html and DBC page design items defined
file_search = dcc.Markdown(children="Working File", className='lbl_working')
working_file = dcc.Upload(id='upload_data', children="Browse or Drag-n-Drop File", className='upload_input',
                          disabled=False, multiple=True)
btn_reset = dbc.Button("Reset", id='btn_reset', disabled=True)
time_slider = dcc.RangeSlider(id='time_slider', step=1, min=0,
                              max=0, value=[0, 0], allowCross=False,
                              pushable=1, className='slider_time')
rad_plot_type = dbc.RadioItems(['Map', 'Antenna'], 'Map', id='plot_type', inline=True)
rad_color = dbc.RadioItems(['DRC', 'SINR', 'RX', 'TX', 'Flight'], 'Flight', id='trace', inline=True)
rad_minute_plot = dbc.RadioItems(['1 minute', '5 minute'], '1 minute', id='minute_plot', inline=True)
check_rx_high = dbc.Checkbox("chk_rx_primary", label="RX Primary (antenna only)", value=False, disabled=True)
check_bts_plot = dbc.Checkbox("chk_bts_plot", label="BTS", value=True, disabled=False)
legend = dcc.Markdown("Legend:", className='lbl_legend')
legend_col_1 = dcc.Markdown("", id='legend_col1', className='legend_col1')
legend_col_2 = dcc.Markdown("", id='legend_col2', className='legend_col2')
legend_col_3 = dcc.Markdown("", id='legend_col3', className='legend_col3')
legend_col_4 = dcc.Markdown("", id='legend_col4', className='legend_col4')
legend_col_5 = dcc.Markdown("", id='legend_col5', className='legend_col5')
lru_label = dcc.Markdown("LRU: ", className='lru_label')
lru_text = dcc.Markdown("", id='lru_text', className='lru_text')

# initial load of geo map
fig_plot = go.Figure(data=go.Scattergeo())
fig_plot.update_layout(
    geo=dict(
        center=dict(lon=-98.6, lat=39.8),
        resolution=50,
        scope='north america',
        lataxis_range=[22, 51], lonaxis_range=[-125, -66],
        showcountries=True,
        showsubunits=True, subunitcolor="#002b36",
        bgcolor='#002b36'),
    autosize=True,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor='#002b36'
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([file_search], width=2),
        dbc.Col([working_file], width=6),
        dbc.Col([btn_reset], width=1)
    ], className='row1'),  # end row 1

    dbc.Row([
        dbc.Col([rad_plot_type], width=3),
        dbc.Col([lru_label], width=1),
        dbc.Col([lru_text], width=2),

    ], className='row2'),  # flight or airborne

    dbc.Row([
        dbc.Col([dcc.Markdown("Trace by:")], width=2),
        dbc.Col([rad_color], width=5),
    ], className='row3'),

    dbc.Row([
        dbc.Col(dcc.Markdown('Plot:'), width=1),
        dbc.Col([rad_minute_plot], width=3),
        dbc.Col([check_bts_plot], width=1),
        dbc.Col([check_rx_high], width=4)
    ], className='row4'),

    dbc.Row([
        # dbc.Col([graph], width=9, className='graph'),
        dbc.Col(dls.Fade(dcc.Graph(figure=fig_plot, id='fig_plot'), color="#435278"),
                width=9, className='graph'),
        dbc.Col([
            dbc.Table([
                html.Tr([html.Th(""), html.Th("Service Minutes"), html.Th("Avg DRC")]),
                html.Tr([html.Td("Hpol"), html.Td("", id="hpol_srv_min"), html.Td("", id="hpol_avg_drc")]),
                html.Tr([html.Td("Vpol"), html.Td("", id="vpol_srv_min"), html.Td("", id="vpol_avg_drc")]),
            ], bordered=True, className='tbl_service'),

            dbc.Table([
                html.Tr([html.Th("Antenna"), html.Th("Port"), html.Th("Xpol")]),
                html.Tr([html.Td("FWD", rowSpan=2), html.Td("Hpol"), html.Td("", id="fwd_hpol")]),
                html.Tr([html.Td("Vpol"), html.Td("", id="fwd_vpol")]),
                html.Tr([html.Td("AFT", rowSpan=2), html.Td("Hpol"), html.Td("", id="aft_hpol")]),
                html.Tr([html.Td("Vpol"), html.Td("", id="aft_vpol")]),
            ], bordered=True, className='tbl_xpol'),
            dbc.Table([
                html.Tr([html.Td("Flight Minutes:"), html.Td("", id="flight_min")])
            ], bordered=True, className='flight_minutes')
        ], width=3),
    ], className='graph_row'),

    dbc.Row([
        dbc.Col([time_slider], id='slider_row', width=9)
    ]),

    dbc.Row([
        dbc.Col([legend], width=1),
        dbc.Col([legend_col_1], width=2),
        dbc.Col([legend_col_2], width=2),
        dbc.Col([legend_col_3], width=2),
        dbc.Col([legend_col_4], width=2),
        dbc.Col([legend_col_5], width=2),
    ], className='legend_row'),

])


# callback when file uploaded
@app.callback(
    Output('upload_data', 'children'),
    Output('upload_data', 'disabled'),
    Output('btn_reset', 'disabled', allow_duplicate=True),
    Output('fig_plot', 'figure', allow_duplicate=True),
    Output('time_slider', 'marks', allow_duplicate=True),
    Output('time_slider', 'min', allow_duplicate=True),
    Output('time_slider', 'max', allow_duplicate=True),
    Output('time_slider', 'value', allow_duplicate=True),
    Output('legend_col1', 'children', allow_duplicate=True),
    Output('legend_col1', 'style', allow_duplicate=True),
    Output('legend_col2', 'children', allow_duplicate=True),
    Output('legend_col2', 'style', allow_duplicate=True),
    Output('legend_col3', 'children', allow_duplicate=True),
    Output('legend_col3', 'style', allow_duplicate=True),
    Output('legend_col4', 'children', allow_duplicate=True),
    Output('legend_col4', 'style', allow_duplicate=True),
    Output('legend_col5', 'children', allow_duplicate=True),
    Output('legend_col5', 'style', allow_duplicate=True),
    Output('lru_text', 'children', allow_duplicate=True),
    Output('hpol_srv_min', 'children', allow_duplicate=True),
    Output('hpol_avg_drc', 'children', allow_duplicate=True),
    Output('vpol_srv_min', 'children', allow_duplicate=True),
    Output('vpol_avg_drc', 'children', allow_duplicate=True),
    Output('fwd_hpol', 'children', allow_duplicate=True),
    Output('fwd_vpol', 'children', allow_duplicate=True),
    Output('aft_hpol', 'children', allow_duplicate=True),
    Output('aft_vpol', 'children', allow_duplicate=True),
    Output('flight_min', 'children', allow_duplicate=True),
    Input('upload_data', 'contents'),
    Input('upload_data', 'filename'),
    Input('plot_type', 'value'),
    Input('trace', 'value'),
    Input('chk_rx_primary', 'value'),
    Input("chk_bts_plot", 'value'),
    Input("minute_plot", 'value'),
    prevent_initial_call=True
 )
def data_file_loaded(list_of_contents, list_of_names, plot, trace, rx_primary, bts_plot, minute_plot):
    global df_static
    global xpol_report
    global range_of_slider
    global legend_html
    global html_style

    if list_of_contents is not None and ctx.triggered_id == 'upload_data' and df_static is False:
        contents = list_of_contents[0]
        filename = list_of_names[0]
        build_dataframe = parse_contents(contents, filename)
        if build_dataframe is True and len(df_static) > 0:
            #return of time_bar [time_start, time_end, marker, minimum, maximum, value]

            time_bar = time_slider_config(minute_plot)
            time_start = 0
            time_end = len(df_static) - 1
            xpol_report.clear()
            xpol_report = xpol_fill(minute_plot, time_start, time_end)
            plot_load = plotter(minute_plot, time_start, time_end, plot, trace, rx_primary, bts_plot)
            range_of_slider = time_bar[5]
            legend_html, html_style = legend_plotter(plot, trace)
            lru = find_lru_sn()
            return (list_of_names, True, False, plot_load, time_bar[2], time_bar[3],
                    time_bar[4], time_bar[5], legend_html[0], html_style[0], legend_html[1], html_style[1],
                    legend_html[2], html_style[2], legend_html[3], html_style[3], legend_html[4], html_style[4], lru,
                    xpol_report[0], xpol_report[1], xpol_report[2], xpol_report[3], xpol_report[4], xpol_report[5],
                    xpol_report[6], xpol_report[7], xpol_report[8])
        elif plot == 'Map':  # file entered was not verified and map is being shown
            plot_load = go.Figure(data=go.Scattergeo())
            plot_load.update_layout(
                geo=dict(
                    resolution=50,
                    scope='north america',
                    showcountries=True,
                    showsubunits=True,
                    subunitcolor="#002b36",
                    bgcolor='#002b36'),
                autosize=True,
                margin=dict(l=2, r=10, t=1, b=1),
                annotations=[
                    {
                        "text": "File is not correct type or format.",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 28}
                    }
                ])
            range_of_slider = [0, 0]
            marker = None
            legend_html.clear()
            html_style.clear()
            for i in range(5):
                legend_html.append("")
                html_style.append({})
            xpol_report.clear()
            for i in range(9):
                xpol_report.append("")
            return ("File wrong type or format", False, True, fig_plot, marker, 0, 0, range_of_slider,
                    legend_html[0], html_style[0], legend_html[1], html_style[1],
                    legend_html[2], html_style[2], legend_html[3], html_style[3], legend_html[4], html_style[4], "",
                    xpol_report[0], xpol_report[1], xpol_report[2], xpol_report[3], xpol_report[4], xpol_report[5],
                    xpol_report[6], xpol_report[7], xpol_report[8])
        else:  # Airborne map being shown and file was not verified
            range_of_slider = [0, 0]
            marker = None
            plot_load = make_subplots(rows=2, cols=2, specs=[[{'type': 'polar'}] * 2] * 2,
                                      subplot_titles=('FWD Aircard 1',
                                                      'FWD Aircard 2',
                                                      'AFT Aircard 1',
                                                      'AFT Aircard 2'))
            plot_load.update_layout(
                title=f'Antenna Plot ',
                showlegend=False,
                autosize=True,
                hoverlabel=dict(
                    font_size=12,
                    namelength=65,
                ),
                annotations=[
                    {
                        "text": "File is not correct type or format.",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 28}
                    }
                ])
            legend_html.clear()
            html_style.clear()
            for i in range(5):
                legend_html.append("")
                html_style.append({})
            xpol_report.clear()
            for i in range(9):
                xpol_report.append("")
            return ("File wrong type or format", False, True, plot_load, marker, 0, 0, range_of_slider,
                    legend_html[0], html_style[0], legend_html[1], html_style[1],
                    legend_html[2], html_style[2], legend_html[3], html_style[3], legend_html[4], html_style[4], "",
                    xpol_report[0], xpol_report[1], xpol_report[2], xpol_report[3], xpol_report[4], xpol_report[5],
                    xpol_report[6], xpol_report[7], xpol_report[8])
    else:
        raise PreventUpdate


# user changes plot type map or antenna plot or BTS Plot checkbox or rx_primary or trace
@app.callback(
    Output('fig_plot', 'figure', allow_duplicate=True),
    Output('chk_rx_primary', 'disabled'),
    Output('chk_bts_plot', 'disabled'),
    Output('legend_col1', 'children', allow_duplicate=True),
    Output('legend_col1', 'style', allow_duplicate=True),
    Output('legend_col2', 'children', allow_duplicate=True),
    Output('legend_col2', 'style', allow_duplicate=True),
    Output('legend_col3', 'children', allow_duplicate=True),
    Output('legend_col3', 'style', allow_duplicate=True),
    Output('legend_col4', 'children', allow_duplicate=True),
    Output('legend_col4', 'style', allow_duplicate=True),
    Output('legend_col5', 'children', allow_duplicate=True),
    Output('legend_col5', 'style', allow_duplicate=True),
    Input('plot_type', 'value'),
    Input('trace', 'value'),
    Input('chk_rx_primary', 'value'),
    Input("chk_bts_plot", 'value'),
    Input("minute_plot", 'value'),
    prevent_initial_call=True,
)
def plot_update(plot, trace, rx_primary, bts_plot, minute_plot):
    global df_static
    global range_of_slider
    global legend_html
    global html_style

    #plot_type, rx_primary, bts_plot changed
    if ((ctx.triggered_id == "plot_type" or ctx.triggered_id == "chk_rx_primary" or ctx.triggered_id == "chk_bts_plot")
            and df_static is not False):
        time_start = range_of_slider[0]
        time_end = range_of_slider[1]
        new_plot = plotter(minute_plot, time_start, time_end, plot, trace, rx_primary, bts_plot)
        if plot == "Antenna":
            rx_primary_visible = False
            bts_plot_visible = True
        else:
            rx_primary_visible = True
            bts_plot_visible = False
        return (new_plot, rx_primary_visible, bts_plot_visible, legend_html[0], html_style[0], legend_html[1],
                html_style[1], legend_html[2], html_style[2], legend_html[3], html_style[3], legend_html[4],
                html_style[4])
    elif ctx.triggered_id == "trace" and df_static is not False:
        # trace type was changed
        time_start = range_of_slider[0]
        time_end = range_of_slider[1]
        legend_html.clear()
        html_style.clear()
        if plot == "Antenna":
            rx_primary_visible = False
            bts_plot_visible = True
        else:
            rx_primary_visible = True
            bts_plot_visible = False
        plot_trace = plotter(minute_plot, time_start, time_end, plot, trace, rx_primary, bts_plot)
        legend_html, html_style = legend_plotter(plot, trace)
        return (plot_trace, rx_primary_visible, bts_plot_visible, legend_html[0], html_style[0], legend_html[1],
                html_style[1], legend_html[2], html_style[2], legend_html[3], html_style[3], legend_html[4],
                html_style[4])
    elif ctx.triggered_id == "minute_plot" and df_static is not False:
        time_start = range_of_slider[0]
        time_end = range_of_slider[1]
        if plot == "Antenna":
            rx_primary_visible = False
            bts_plot_visible = True
        else:
            rx_primary_visible = True
            bts_plot_visible = False
        minute_changed = plotter(minute_plot, time_start, time_end, plot, trace, rx_primary, bts_plot)
        return (minute_changed, rx_primary_visible, bts_plot_visible, legend_html[0], html_style[0], legend_html[1],
                html_style[1], legend_html[2], html_style[2], legend_html[3], html_style[3], legend_html[4],
                html_style[4])
    else:
        raise PreventUpdate


#time of view has changed
@app.callback(
    Output('fig_plot', 'figure', allow_duplicate=True),
    Output('hpol_srv_min', 'children', allow_duplicate=True),
    Output('hpol_avg_drc', 'children', allow_duplicate=True),
    Output('vpol_srv_min', 'children', allow_duplicate=True),
    Output('vpol_avg_drc', 'children', allow_duplicate=True),
    Output('fwd_hpol', 'children', allow_duplicate=True),
    Output('fwd_vpol', 'children', allow_duplicate=True),
    Output('aft_hpol', 'children', allow_duplicate=True),
    Output('aft_vpol', 'children', allow_duplicate=True),
    Output('flight_min', 'children', allow_duplicate=True),
    Input('time_slider', 'value'),
    Input('plot_type', 'value'),
    Input('trace', 'value'),
    Input('chk_rx_primary', 'value'),
    Input("chk_bts_plot", 'value'),
    Input("minute_plot", 'value'),
    prevent_initial_call=True,
)
def time_update(slider_range, plot, trace, rx_primary, bts_plot, minute_plot):
    global range_of_slider
    global xpol_report
    global df_static

    if (ctx.triggered_id == 'time_slider' and slider_range != range_of_slider and range_of_slider is not None and
            df_static is not False):

        time_start = slider_range[0]
        time_end = slider_range[1]
        xpol_report.clear()
        xpol_report = xpol_fill(minute_plot, time_start, time_end)
        time_plot = plotter(minute_plot, time_start, time_end, plot, trace, rx_primary, bts_plot)
        range_of_slider.clear()
        range_of_slider = slider_range

        return (time_plot, xpol_report[0], xpol_report[1], xpol_report[2], xpol_report[3], xpol_report[4],
                xpol_report[5], xpol_report[6], xpol_report[7], xpol_report[8])
    else:
        raise PreventUpdate


#reset all back to original
@app.callback(  # reset all callback
    Output('fig_plot', 'figure', allow_duplicate=True),
    Output('upload_data', 'children', allow_duplicate=True),
    Output('upload_data', 'disabled', allow_duplicate=True),
    Output('btn_reset', 'disabled', allow_duplicate=True),
    Output('chk_rx_primary', 'value', allow_duplicate=True),
    Output('plot_type', 'value', allow_duplicate=True),
    Output('upload_data', 'contents'),
    Output('time_slider', 'marks', allow_duplicate=True),
    Output('time_slider', 'min', allow_duplicate=True),
    Output('time_slider', 'max', allow_duplicate=True),
    Output('time_slider', 'value', allow_duplicate=True),
    Output('lru_text', 'children'),
    Output('hpol_srv_min', 'children', allow_duplicate=True),
    Output('hpol_avg_drc', 'children', allow_duplicate=True),
    Output('vpol_srv_min', 'children', allow_duplicate=True),
    Output('vpol_avg_drc', 'children', allow_duplicate=True),
    Output('fwd_hpol', 'children', allow_duplicate=True),
    Output('fwd_vpol', 'children', allow_duplicate=True),
    Output('aft_hpol', 'children', allow_duplicate=True),
    Output('aft_vpol', 'children', allow_duplicate=True),
    Output('flight_min', 'children', allow_duplicate=True),
    Output('chk_bts_plot', 'value'),
    Output('minute_plot', 'value'),
    Output('legend_col1', 'children', allow_duplicate=True),
    Output('legend_col1', 'style', allow_duplicate=True),
    Output('legend_col2', 'children', allow_duplicate=True),
    Output('legend_col2', 'style', allow_duplicate=True),
    Output('legend_col3', 'children', allow_duplicate=True),
    Output('legend_col3', 'style', allow_duplicate=True),
    Output('legend_col4', 'children', allow_duplicate=True),
    Output('legend_col4', 'style', allow_duplicate=True),
    Output('legend_col5', 'children', allow_duplicate=True),
    Output('legend_col5', 'style', allow_duplicate=True),
    Input('btn_reset', 'n_clicks'),
    prevent_initial_call=True,
)
def reset_page(reset_pressed):
    global df_static
    global range_of_slider
    global xpol_report
    global legend_html
    global html_style

    del df_static
    df_static = False
    gc.collect()
    time_start = 0
    time_end = 0
    xpol_report.clear()
    time_bar = time_slider_config('1 minute')
    range_of_slider = [0, 0]
    reset_plot = go.Figure(data=go.Scattergeo())
    reset_plot.update_layout(
        geo=dict(
            center=dict(lon=-98.6, lat=39.8),
            resolution=50,
            scope='north america',
            lataxis_range=[22, 51], lonaxis_range=[-125, -66],
            showcountries=True,
            showsubunits=True, subunitcolor="#002b36",
            bgcolor='#002b36'),
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    legend_html.clear()
    html_style.clear()
    legend_html, html_style = legend_plotter('empty', 'Flight')
    for i in range(9):
        xpol_report.append("")


    return (reset_plot, "Browse or Drag-n-Drop File", False, True, False, 'Map', None, time_bar[2], time_bar[3],
            time_bar[4], time_bar[5], "", xpol_report[0], xpol_report[1], xpol_report[2], xpol_report[3],
            xpol_report[4], xpol_report[5], xpol_report[6], xpol_report[7], xpol_report[8], True, '1 minute',
            legend_html[0], html_style[0], legend_html[1], html_style[1], legend_html[2], html_style[2], legend_html[3],
            html_style[3], legend_html[4], html_style[4])


if __name__ == '__main__':
    #app.run_server(host='0.0.0.0') #for deployment
    df_static = False
    range_of_slider = None
    legend_html = []
    html_style = []
    xpol_report = []
    app.run_server(debug=True)  #run local server
