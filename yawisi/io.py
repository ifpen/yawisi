import os
import numpy as np

from struct import pack, unpack
import time

from yawisi.wind_field import WindField
from yawisi.parameters import SimulationParameters
from yawisi.locations import Grid
from yawisi.wind import Wind

from yawisi import __version__

_TS_BIN_FMT = '<h4l12fl'  # TurbSim binary format


def to_bts(wind_field: WindField, path, uzhub=None, periodic=True):
    """yawisi wind_field to TurbSim-style binary file
    Code modified based on `turbsim` in PyTurbSim:
        https://github.com/lkilcher/pyTurbSim/blob/master/pyts/io/write.py

    Notes
    -----
    * The turbulence must have been generated on a y-z grid (Locations = Grid).
    """
    if path.endswith('.bts'):  # remove file extension, will be added back later
        path = path[:-4]
    # format-specific constants
    intmin = -32768  # minimum integer
    intrng = 65535  # range of integers
    # calculate intermediate parameters
    y = np.sort(np.unique(wind_field.locations.y_array()))
    z = np.sort(np.unique(wind_field.locations.z_array()))
    ny = y.size  # no. of y points in grid
    nz = z.size  # no. of z points in grif
    nt = wind_field.params.n_samples # no. of time steps
    if y.size == 1:
        dy = 0
    else:
        dy = np.mean(y[1:] - y[:-1])  # hopefully will reduce possible errors
    if z.size == 1:
        dz = 0
    else:
        dz = np.mean(z[1:] - z[:-1])  # hopefully will reduce possible errors
    dt = wind_field.params.sample_time  # time step
    if uzhub is None:  # default is center of grid
        zhub = z[z.size // 2]  # halfway up
        uhub = wind_field.get_umean()  # mean of center of grid
    else:
        uhub, zhub = uzhub
    
    # convert pyconturb dataframe to pyturbsim format (3 x ny x nz x nt)
    ts = wind_field.get_uvwt()
    
    # initialize output arrays
    u_off = np.empty((3), dtype=np.float32)  # offsets of each time series
    u_scl = np.empty((3), dtype=np.float32)  # scales of each time series
    desc_str = 'generated by YaWiSi v%s, %s.' % (
        __version__,
        time.strftime('%b %d, %Y, %H:%M (%Z)', time.localtime()))  # description
    # calculate the scales and offsets of each time series
    out    = np.empty(ts.shape, dtype=np.int16)
    for k in range(3):
        all_min, all_max = ts[k].min(), ts[k].max()
        if all_min == all_max:
            u_scl[k] = 1
        else:
            u_scl[k] = intrng / (all_max-all_min)
        u_off[k] = intmin - u_scl[k] * all_min
        out[k] = (ts[k] * u_scl[k] + u_off[k]).astype(np.int16)
    with open(path + '.bts', 'wb') as fl:
        # write the header
        fl.write(pack(_TS_BIN_FMT,
                      [7, 8][periodic],  # 7 is not periodic, 8 is periodic
                      nz,
                      ny,
                      0,  # assuming 0 tower points below grid
                      nt,
                      dz,
                      dy,
                      dt,
                      uhub,
                      zhub,
                      z[0],
                      u_scl[0],
                      u_off[0],
                      u_scl[1],
                      u_off[1],
                      u_scl[2],
                      u_off[2],
                      len(desc_str)))
        fl.write(desc_str.encode(encoding='UTF-8'))
      
        # The indexes vary in the following order:
        # component (fastest), y-index, z-index, time (slowest).
        for it in np.arange(nt):
            fl.write(out[:,it,:,:].tobytes(order='F'))
        

def from_bts(filename, tdecimals=8) -> WindField:
    """ read BTS file, u  (3 x nt x ny x nz)
    """

    if not filename:
        raise Exception('No filename provided')
    if not os.path.isfile(filename):
        raise OSError(2,'File not found:',filename)
    if os.stat(filename).st_size == 0:
        raise Exception('File is empty:',filename)

    scl = np.zeros(3, np.float32); off = np.zeros(3, np.float32)
    with open(filename, mode='rb') as f:            
        # Reading header info
        ID, nz, ny, nTwr, nt                      = unpack('<h4l', f.read(2+4*4))
        dz, dy, dt, uHub, zHub, zBottom           = unpack('<6f' , f.read(6*4)  )
        scl[0],off[0],scl[1],off[1],scl[2],off[2] = unpack('<6f' , f.read(6*4))
        nChar, = unpack('<l',  f.read(4))
        info = (f.read(nChar)).decode()
       
        # Reading turbulence field
        u    = np.zeros((3,nt,ny,nz))
        # For loop on time (acts as buffer reading, and only possible way when nTwr>0)
        for it in range(nt):
            Buffer = np.frombuffer(f.read(2*3*ny*nz), dtype=np.int16).astype(np.float32).reshape([3, ny, nz], order='F')
            u[:,it,:,:]=Buffer
            if nTwr > 0:
                np.frombuffer(f.read(2*3*nTwr), dtype=np.int16)
        u -= off[:, None, None, None]
        u /= scl[:, None, None, None]
    
    
    y    = np.arange(ny)*dy 
    y   -= np.mean(y) # y always centered on 0
    z    = np.arange(nz)*dz +zBottom
 
    params = SimulationParameters(None)
    params.grid_height = z[-1] - z[0]
    params.grid_width = y[-1] - y[0]
    params.grid_length = int(y.shape[0])
    params.wind_mean = uHub
    params.n_samples = nt 
    params.sample_time = np.round(dt, tdecimals) # dt is stored in single precision in the TurbSim output
    print(params)

    wind_field = WindField(params)
    wind_field.info = info
    grid: Grid = wind_field.locations
    grid.assign(y, z)

    for i_pt in range(len(grid)):
        wind = Wind(params)
        i, j = grid.coords(i_pt)
        wind.wind_values = np.transpose(u[:, :, i, j])
        wind_field.wind.append(wind)
    
    return wind_field
    
