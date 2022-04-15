# Pointers to columns in the dataset

columns= {
    'date':'DATE (YYYY/MM/DD)',
    'time':'MST',
    'IR':'Global CMP22 (vent/cor) [W/m^2]',
    'WB':'Tower Wet Bulb Temp [deg C]',
    'normIR':'Direct sNIP [W/m^2]', 
    'angle':'Azimuth Angle [degrees]',
    # 'DB':'Tower Dry Bulb Temp [deg C]', 
    'DP':'Tower Dew Point Temp [deg C]',
    'RH':'Tower RH [%]', 
    'clouds':'Total Cloud Cover [%]',
    'WS':'Peak Wind Speed @ 6ft [m/s]', 
    'WD':'Avg Wind Direction @ 6ft [deg from N]',
    'P':'Station Pressure [mBar]', 
    'precip':'Precipitation (Accumulated) [mm]',
    'snow':'Snow Depth [cm]', 
    'moisture':'Moisture', 
    'alpha':'Albedo (CMP11)'
    }



experiments = {
    'all': {
        'columns':list(columns.values()),
        'negone':False,  # normalize between [0,1]
        'derivative':None, 
        'activation':'relu',
        'type':'train',
        'hours':2,
        'dropnight':True
    }
}