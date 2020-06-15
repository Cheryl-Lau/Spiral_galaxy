# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt 
import collections 
from scipy import interpolate
from scipy import optimize
import random as rd 
import vpython as vp 
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 


''' 
Spiral Galaxy with Density Wave Theory
'''

class OrbitingStar:
    '''
    Generate a star within the given cell boundary, following a well-defined elliptical orbit around 
    the galactic center, allowing the spiral structure to be maintained. 
    '''
    def __init__(self, cell_bound, b_phi_func, min_phi, max_phi, radius, rgb_color, opacity, rc_func, z_pos):
        self.cell_bound = cell_bound  # boundaries of the cell containing this star
        self.b_phi_func = b_phi_func  # callable function of axis_b against tilt_phi of the ellipses in spiral pattern
        self.min_phi = min_phi        # minimum phi value to be searched in solving b-phi equations 
        self.max_phi = max_phi        # maximum phi value to be searched 
        self.radius = radius          # star size 
        self.rgb_color = rgb_color    # star colour in RGB vector form 
        self.opacity = opacity        # star opacity 
        self.rc_func = rc_func        # callable function of galactic rotation curve 
        self.z_pos = z_pos            # z-position of star (layer of galactic disc)
        
        # Initialize position 
        self.position = init_star_position(self.cell_bound, self.z_pos)
        
        # Current theta0, measured from x-axis
        self.theta0 = np.arctan(self.position.y/self.position.x)
        if self.position.x < 0:
            self.theta0 += np.pi 
        
        # Solve equations to find phi and b of this star to define its orbit 
        self.err_flag = False  # marks whether or not a solution was found 
        try:
            self.phi, self.axis_b = solve_b_phi_eqn(self.position, self.b_phi_func, self.min_phi, self.max_phi)
        except:
            self.err_flag = True
            
        # Proceed only if a b-phi solution was successfully found 
        if self.err_flag == False:           
            # Draw vpython sphere
            self.star_obj = vp.sphere(pos=self.position,radius=self.radius,color=self.rgb_color,opacity=self.opacity)        
            # Initialize tangential velocity 
            self.velocity = init_star_velocity(self.position, self.rc_func)
            
            
    def motion(self):
        ''' Motion of individual star. Moves at a constant tangential velocity. '''
        if self.err_flag == False:
            
            # Update theta0 of the star 
            self.theta0 -= self.velocity/self.axis_b*time_scale  

            # theta0 = theta + phi angles in ellipse parametric eqn, so
            self.theta = self.theta0 - self.phi
            
            # Calculate updated position 
            x, y = ellipse(self.axis_b, self.phi, self.theta)
            self.position = vp.vector(x,y,self.z_pos)
            
            # Upate vpython object position
            self.star_obj.pos = self.position
            
            
def init_star_position(cell_bound, z_pos):
    ''' Generate star at random position within the cell boundary '''
    # Uniform random x-coord 
    x_coord = rd.uniform(*cell_bound[0])  
    y_coord = rd.uniform(*cell_bound[1])  
    position = vp.vector(x_coord,y_coord,z_pos)
    
    return position 


def solve_b_phi_eqn(position, b_phi_func, min_phi, max_phi):
    '''
    To find the axis_b and phi of the star's orbit with the given xy-position.
    Two equations describe the relation between axis_b and tilt_phi in this system:
        1. equation of a tilted ellipse (with x and y known, and axis_a is a func of axis_b; write b as a func of phi)
        2. the b-phi interpolated relation derived from the spiral structure
    Solving the two equations simultaneously gives the phi and b for a star on the spiral at this position
    '''
    # Function of difference in axis_b calculated with the two equations
    diff_func = lambda phi: ((1-eccen**2)*(position.x*np.cos(phi) + position.y*np.sin(phi))**2 + \
                  (position.x*np.sin(phi) - position.y*np.cos(phi))**2)**(1/2) - b_phi_func(phi)
    
    # Find root (i.e. phi of intersection pt)
    try:
        solution = optimize.root_scalar(diff_func, bracket=[min_phi,max_phi], method='brentq') 
    except:  
        raise ValueError # no root found     
    else:    # proceed 
        phi_sol = solution.root
        b_sol = ((1-eccen**2)*(position.x*np.cos(phi_sol) + position.y*np.sin(phi_sol))**2 + \
                 (position.x*np.sin(phi_sol) - position.y*np.cos(phi_sol))**2)**(1/2)
        
        return phi_sol, b_sol
    

def init_star_velocity(position, rc_func):
    ''' 
    Get star's orbiting velocity from the interpolated Galactic rotation curve function. 
    Otherwise, if the input radius value is beyond the interpolated function, 
    find the closest point in radius data, and use the corresponding velocity.
    '''
    # Orbital radius of star on xy-plane 
    star_radius = vp.mag(vp.vector(position.x,position.y,0) - vp.vector(0,0,0))
    
    try:
        # call RC interpolated function with star radius as input 
        velocity = rc_func(star_radius)
    except:
        radius_data, velocity_data = np.loadtxt(rc_filename,unpack=True)  
        radius_data = radius_data*3.086e+16  # kpc to km 
        # get point where difference between input and data is minimized 
        diff_array = abs(radius_data - star_radius)
        min_index = np.where(diff_array == min(diff_array))
        velocity = velocity_data[min_index]
        
    return velocity 
    

''' ###### Functions for drawing ellipses to generate a spiral structure - '''


def ellipse(axis_b, tilt_phi, theta):
    '''
    A tilted ellipse defining the orbits in a galaxy;
    Ellipses have varying radius and tilt-angle in order to form a spiral structure in density.
    Parameters of the ellipse are all functions of axis_b (semi-minor axis), including tilt_phi angle
    (which will be solved later).
    Angle theta defines a single point on the ellipse; measured from axis_b not x-axis. 
    Returns xy-coord of the point. 
    '''
    # Calculate aixs_a 
    axis_a = np.sqrt(axis_b**2/(1-eccen**2))
    # Parametric equation of ellipse 
    x = axis_a*np.cos(theta)*np.cos(tilt_phi) - axis_b*np.sin(theta)*np.sin(tilt_phi)
    y = axis_a*np.cos(theta)*np.sin(tilt_phi) + axis_b*np.sin(theta)*np.cos(tilt_phi)
    
    return x, y
    

def density_wave_radius(galactic_radius, line_sep_const):
    '''
    Surface brightness of a galaxy is described by the exponential law 
    I(R) = I0 * exp(-R/R_D)  where R_D is the characteristic scale length.
    Brightness is proportional to the density of ellipse lines; hence inversely proportional to
    seperations between the ellipses. 
    Compute an array of radius (axis_b) values of the ellipses such that the line seperation 
    grows exponentially.
    '''
    # init a uniform radius array; also defines the number of ellipses to be generated
    radius_array = np.linspace(0,galactic_radius,num_ellipse)
    # define characteristic scale length R_D
    char_radius = radius_array[int(len(radius_array)/5.)] # estimated half-life 
    # line separation is inversely proportional to brightness, so
    line_sep_array = line_sep_const * np.exp(radius_array/char_radius)
    # build up a new radius array in which the separation between each pt grows exponentially 
    exp_radius_array = np.r_[radius_array[0],line_sep_array].cumsum()

    return exp_radius_array
    

def sep_const_search(galactic_radius):
    '''
    Looping the density_wave_radius function. 
    Use bisection method to search for the most appropriate line_sep_const value such that 
    the max value in exp_radius_array is 1.1x (slightly) bigger than the real galactic radius.
    '''
    const_min = line_sep_const_min 
    const_max = line_sep_const_max
    
    exp_radius_array = np.array([]) # init variable for checking exception 
    
    while const_max - const_min > 1e5:  # define tolerance
        # run with test value of const
        const = (const_max + const_min)/2.
        exp_radius_array = density_wave_radius(galactic_radius, const)
        
        # check the max value against galactic radius
        if exp_radius_array[-1] > galactic_radius*1.1:
            const_max = const
        if exp_radius_array[-1] < galactic_radius*1.1:   
            const_min = const
             
    # output the final exp_radius_array
    if exp_radius_array.size:  
        return exp_radius_array
    else:
        raise Exception('Range of line_sep_const cannot be smaller than the tolerance level')


def init_ellipses(exp_radius_array):
    ''' 
    Generate a collection of xy-coords of all ellipses, with their corresponding axis_b and 
    tilt_phi for defining star orbits later on 
    '''
    # theta of ellipse goes through one revolution 
    theta_array = np.linspace(0,2*np.pi,1000)
    
    ell_x_coord = []
    ell_y_coord = []
    axis_b_list = []
    tilt_phi_list = []
    for i,axis_b in enumerate(exp_radius_array):  
        for theta in theta_array:
            # Define tilt-angle phi in terms of ellipse no. 
            tilt_phi = shift_deg *np.pi/180 * i 

            x, y = ellipse(axis_b, tilt_phi, theta)  
            
            axis_b_list.append(axis_b)
            tilt_phi_list.append(tilt_phi)
            ell_x_coord.append(x)
            ell_y_coord.append(y)  
            
    return ell_x_coord, ell_y_coord, axis_b_list, tilt_phi_list


''' ###### Functions for making a mathematical function axis_b(tilt_phi) - '''


def exp_interp1d(x_array, y_array):
    ''' Used in axis_b_phi_interp. Interpolates any x- and y-dataset of an exponentially-ish growing curve '''
    # Curvy lines on original graph can turn into straight lines on log transformed graphs
    # transform the y-axis into log scale
    log_y_array = np.log10(y_array)
    # treat as stright-ish line to do linear interpolation 
    lin_interp = interpolate.interp1d(x_array, log_y_array, kind='linear')
    # create a function which can take an input x, stick into the linear interped function
    # get the output log_y and convert back to y 
    exp_interp_func = lambda input_x: np.power(10.,lin_interp(input_x))
    
    return exp_interp_func   


def axis_b_phi_interp(tilt_phi_list, axis_b_list):
    ''' 
    Interpolate the discrete b and phi values of the ellipses to get a function which discribes 
    the relation between b and phi on the spiral structure. 
    This function is later used to solve (along with the ellipse eqn) the b and phi of 
    any given xy-point, i.e. the orbit of individual stars. 
    '''
    # Plot raw b and phi of the ellipses 
    fig2 = plt.figure(figsize=(8,5)) 
    ax2 = fig2.add_subplot(111)
    ax2.plot(tilt_phi_list,axis_b_list,'o',color='orange')
    
    # The b-phi discrete points shows an exponential-like curve 
    # Generate a function describing this curve 
    b_phi_func = exp_interp1d(tilt_phi_list, axis_b_list)
    
    # test plot interpolation function 
    test_phi_array = np.linspace(0,tilt_phi_list[-1],1000)
    test_axis_b_array = b_phi_func(test_phi_array)
    ax2.plot(test_phi_array,test_axis_b_array,'--',color='blue')
    plt.show()
    
    return b_phi_func
    

''' ###### Functions for putting stars onto the spiral structure - '''


def cell_boundaries(grid_min, grid_max, num_cell):
    ''' Used in init_grid. Defines the xy boundaries of each cell. '''
    boundaries = np.linspace(grid_min,grid_max,num_cell)
    cell_bounds = []
    for i in range(int(num_cell)-1):
        cell_min = boundaries[i]
        cell_max = boundaries[i+1]
        cell_bounds.append((cell_min,cell_max))
        
    return cell_bounds     # ((0,10),(10,20),(20,30)....number of cells)
        

def find_cell(x_value, y_value, cell_bounds_x, cell_bounds_y):
    ''' Used in init_grid. Reads a data point and assign the right cell boundary to it. '''
    for cell_x_bound in cell_bounds_x:
        if x_value >= cell_x_bound[0] and x_value < cell_x_bound[1]:
            x_bound = cell_x_bound 
    for cell_y_bound in cell_bounds_y:
        if y_value >= cell_y_bound[0] and y_value < cell_y_bound[1]:
            y_bound = cell_y_bound 

    if x_bound and y_bound:  # check empty result
        cell_bound = (x_bound,y_bound)
        return cell_bound     # ((x_min,x_max),(y_min,y_max))
    else:
        raise Exception('Grid dimensions unable to cover entire galactic disc.')


def init_grid(ell_x_coord, ell_y_coord, model_galactic_radius, ax1):
    '''
    Superimpose a 2D grid onto the density wave spiral diagram.
    For each grid cell, count the number of ell_xy_coord points in it.
    To be used for generating the stars' location. 
    '''
    # dimensions of the square grid on the disc 
    grid_min = -model_galactic_radius*1.15
    grid_max = model_galactic_radius*1.15
    
    # list of xy boundaries of all cells 
    cell_bounds_x = cell_boundaries(grid_min, grid_max, num_cell)
    cell_bounds_y = cell_boundaries(grid_min, grid_max, num_cell)
    
    # plot grid onto spiral diagram
    for i in range(len(cell_bounds_x)):
        ax1.axhline(cell_bounds_y[i][0],grid_min,grid_max)
        ax1.axvline(cell_bounds_x[i][0],grid_min,grid_max)

    # a list of cell boundary of each data point 
    cell_bounds_list = []
    for x_value, y_value in zip(ell_x_coord, ell_y_coord): 
        cell_bound = find_cell(x_value, y_value, cell_bounds_x, cell_bounds_y)
        cell_bounds_list.append(cell_bound)

    # for each cell bound, count how many times a data point was assigned to it 
    cell_count_dict = collections.Counter(cell_bounds_list)
    
    return cell_count_dict   # {(x_min1,x_max1),(y_min1,y_max1): 3, (x_min2,x_max2),(y_min2,y_max2): 5, ...}
    

def init_white_stars(cell_count_dict, b_phi_func, min_phi, max_phi, rc_func):
    ''' 
    Generate individual stars of a particular kind. Here, the white stars. 
    At grid cells where lots of ellipse lines pass through, those are high-density regions. 
    The number of stars in each cell is proportional to the number of points (line). 
    '''
    white_star_list = []
    # Loop through each cell in dict
    for cell_bound, count in cell_count_dict.items():
        
        # Define relation between number of stars and cell count
        num_star = num_star_const_white * count  
        
        # Set radius and opacity
        radius = 1.5e15
        opacity = 1.  # max 1
        # Set colour as rgb vector  
        vp_rgb_color = (1/255)*vp.vector(255,255,255)  # max 1 in vpython system
        # Set z-coord of the stars ('layer' of galactic disc)
        z_pos = 1.

        # Generate individual stars 
        for i in range(int(num_star)):
            # init star object 
            star = OrbitingStar(cell_bound, b_phi_func, min_phi, max_phi, radius, vp_rgb_color, opacity, rc_func, z_pos)
            white_star_list.append(star)
       
    return white_star_list 
        

def init_red_stars(cell_count_dict, b_phi_func, min_phi, max_phi, rc_func):
    ''' A few red stars at high-density regions '''
    red_star_list = []
    for cell_bound, count in cell_count_dict.items():
        if count > count_filter_red[0] and count < count_filter_red[1]:
            num_star = num_star_const_red * count**2  
            radius = 2e15
            opacity = .5  # max 1
            vp_rgb_color = (1/255)*vp.vector(242,23,23)  
            z_pos = 1e16
            for i in range(int(num_star)):
                star = OrbitingStar(cell_bound, b_phi_func, min_phi, max_phi, radius, vp_rgb_color, opacity, rc_func, z_pos)
                red_star_list.append(star)
       
    return red_star_list 


def init_blue_stars(cell_count_dict, b_phi_func, min_phi, max_phi, rc_func):
    ''' Blue stars at outer regions '''
    blue_star_list = []
    for cell_bound, count in cell_count_dict.items():
        if count > count_filter_blue[0] and count < count_filter_blue[1]:
            num_star = num_star_const_blue * count 
            radius = 2e15
            opacity = .5  # max 1
            vp_rgb_color = (1/255)*vp.vector(100,105,238)  
            z_pos = -1e16
            for i in range(int(num_star)):
                star = OrbitingStar(cell_bound, b_phi_func, min_phi, max_phi, radius, vp_rgb_color, opacity, rc_func, z_pos)
                blue_star_list.append(star)
       
    return blue_star_list 


def init_yellow_stars(cell_count_dict, b_phi_func, min_phi, max_phi, rc_func):
    ''' Yellow stars near the center '''
    yellow_star_list = []
    for cell_bound, count in cell_count_dict.items():
        if count > count_filter_yellow[0] and count < count_filter_yellow[1]: 
            num_star = num_star_const_yellow * count 
            radius = 1e15
            opacity = .8  # max 1
            vp_rgb_color = (1/255)*vp.vector(255,255,168)  
            z_pos = 1e16
            for i in range(int(num_star)):
                star = OrbitingStar(cell_bound, b_phi_func, min_phi, max_phi, radius, vp_rgb_color, opacity, rc_func, z_pos)
                yellow_star_list.append(star)
       
    return yellow_star_list 


#######################################################################################################################

def init_system(radius_data, velocity_data, rc_func):
    '''
    Main of initializing all galaxy objects 
    '''
    # Radius of galaxy
    galactic_radius = radius_data[-1]
    print('galactic radius: {:.2E} km'.format(galactic_radius))
    
    # Compute axis_b of the ellipses 
    exp_radius_array = sep_const_search(galactic_radius)
    print('galactic radius of model: {:.2E} km'.format(exp_radius_array[-1]))
    
    # Calculate the coord, axis_b and tilt_phi of all ellipses
    ell_x_coord, ell_y_coord, axis_b_list, tilt_phi_list = init_ellipses(exp_radius_array)

    # Plot density wave spiral diagram
    fig1 = plt.figure(figsize=(7,7)) 
    ax1 = fig1.add_subplot(111)
    ax1.plot(ell_x_coord,ell_y_coord,'.',markersize=0.5) 
    plt.show()
    
    # Find the function of axis_b-tilt_phi relation 
    b_phi_func = axis_b_phi_interp(tilt_phi_list, axis_b_list)
        
    # Do a 2D binning to find number of xy-points in each grid cell 
    cell_count_dict = init_grid(ell_x_coord, ell_y_coord, exp_radius_array[-1], ax1)

    # Max value of tilt_phi to be searched when solving eqn
    min_phi = tilt_phi_list[0]
    max_phi = tilt_phi_list[-1]
    
    star_list = []  # list of moving stars' objects     
    # Generate individual stars 
    if show_white_star == True:
        white_star_list = init_white_stars(cell_count_dict, b_phi_func, min_phi, max_phi, rc_func)   
        star_list.extend(white_star_list)
    if show_red_star == True:
        red_star_list = init_red_stars(cell_count_dict, b_phi_func, min_phi, max_phi, rc_func)
        star_list.extend(red_star_list)
    if show_blue_star == True:
        init_blue_stars(cell_count_dict, b_phi_func, min_phi, max_phi, rc_func)   # stationary 
    if show_yellow_star == True:
        yellow_star_list = init_yellow_stars(cell_count_dict, b_phi_func, min_phi, max_phi, rc_func)
        star_list.extend(yellow_star_list)

    return star_list


def main():

    scene = vp.canvas(width=700,height=700,center=vp.vector(0,0,0))
    scene.autoscale = False

    # Read galactic rotation file 
    radius_data, velocity_data = np.loadtxt(rc_filename,unpack=True)
    radius_data = radius_data*3.086e+16  # kpc to km 
    
    # Interpolate the rotation curve; callable function
    rc_func = interpolate.interp1d(radius_data, velocity_data)
    
    # Plot RC data
    fig3 = plt.figure(figsize=(10,5)) 
    ax3 = fig3.add_subplot(111)
    ax3.plot(radius_data, velocity_data,'o',color='blue')
    # Plot RC function 
    rc_func_x = np.linspace(radius_data[0],radius_data[-1],1000)
    rc_func_y = rc_func(rc_func_x)
    ax3.plot(rc_func_x, rc_func_y,'-',color='red')
    plt.show()
    
    scene.camera.pos = vp.vector(0,0,radius_data[-1]*1.5)
    scene.camera.axis = vp.vector(vp.vector(0,0,0) - scene.camera.pos)  
    vp.local_light(pos=vp.vector(0,0,0),color=vp.color.blue)  

    # Initialize all objects 
    star_list = init_system(radius_data, velocity_data, rc_func)
    

    ''' Animation '''
    
    if run_animation == True:       
        i = 0
        while True:
            vp.rate(1e10)
            
            for star in star_list:
                star.motion()
    
            if i%100 == 0:
                print('Star positions update completed - Running next iteration')          
            i += 1



''' Input Settings '''

# Rotation curve datafile
# NCG 4192 from http://www.ioa.s.u-tokyo.ac.jp/~sofue/virgo/virgo2/rcdat/
rc_filename = 'rotation_curve.dat'  # R:kpc, v:km/s

# Fast forward animation
time_scale = 3e13

#### Spiral structure controls - 

# Number of ellipses
num_ellipse = 100
# Eccentricity of the ellipses
eccen = 0.4
# Angular offset in orientation to shift per ellipse
shift_deg = 17.

# Constant of proportionality between exp of radius and the separations between each ellipse
# Define a range of line_sep_const to be searched 
line_sep_const_min = 1e14
line_sep_const_max = 1e15

# Superimposing a grid onto the ellipses to init the locations of the stars
# Number of cells on one side of the square grid
num_cell = 70  

# Run vpython animation (May switch off while adjusting the spiral structure) 
run_animation = True

#### Star controls - 
# set consant of proportionality between number of stars and number of points in each cell
# and which cells (filtered by its number of points) to put the stars into 
# [requires manual adjustment based on the output image]

show_white_star = True
num_star_const_white = 0.3  

show_red_star = True
num_star_const_red = 3e-4
count_filter_red = [0,1e2]   # [min_count,max_count]

show_blue_star = True 
num_star_const_blue = 0.1
count_filter_blue = [0,50]

show_yellow_star = True
num_star_const_yellow = 0.15
count_filter_yellow = [1.5e2,6e2]



if __name__ == '__main__':
    main()













