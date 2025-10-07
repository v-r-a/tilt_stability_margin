# Copyright (c) 2025 Vyankatesh Ashtekar
# All rights reserved.
# This code is provided for academic review only.
# No permission is granted for reuse, modification, or redistribution.


# Note that the limits on forces and moments are not enforced in the demo utilities and the ZMP axis may be go outside the foot supports. Values of forces/moments corresponding to such cases should be assumed as infeasible.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, Slider

def Rz(a):
    Rmat = np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a),  np.cos(a), 0],
                     [0,          0,         1]])
    return Rmat

# Drawings
def old_proj(c: np.ndarray,
             p0: np.ndarray,
             vhp_n: np.ndarray,
             vhp_o: np.ndarray):
    # Gives point intersecting the VHP plane---an extension of COM-P0 line

    k = np.dot(vhp_o - c, vhp_n)/np.dot(p0-c, vhp_n)
    p = c + k * (p0 - c)

    # Plot the line segment: Xplot, Yplot, Zplot
    Xplot = np.array([c[0], p[0]])
    Yplot = np.array([c[1], p[1]])
    Zplot = np.array([c[2], p[2]])
    return Xplot, Yplot, Zplot

def new_proj(com: np.ndarray, 
             mass, 
             me_mag, me_phi, 
             p0, 
             vhp_p, vhp_n):
    # Draws tangent to the sphere---new method
    # The sphere is based on assumption of static equilibrium

    # External moment vector
    me = np.array([me_mag*np.cos(me_phi), me_mag*np.sin(me_phi), 0]) # xy plane is horizontal
    M = me

    # Compute the radius
    a = np.linalg.norm(M / (mass* 9.81))

    # Unit vector along the applied moment
    if np.linalg.norm(M) > 1e-6:
        mhat = M / np.linalg.norm(M)
        # Two basis vectors orthogonal to mhat
        u = np.array([-mhat[1], mhat[0], 0])
        u /= np.linalg.norm(u)
        v = np.cross(mhat, u)
        v /= np.linalg.norm(v)

        # Solve for the tangent point on sphere Acos(th) + Bsin(th) + C =0
        A = np.dot(com-p0,u)
        B = np.dot(com-p0,v)
        C = a

        # Quadratic in half-tangent theta:  (C-A)t^2 + 2Bt + (C+A) = 0
        t1 = -B + np.sqrt(B**2 - (C**2-A**2))
        t1 /= (C-A)
        t2 = -B - np.sqrt(B**2 - (C**2-A**2))
        t2 /= (C-A)

        th1 = 2 * np.arctan(t1)
        th2 = 2 * np.arctan(t2)

        # Possible centers of projection
        Pc1 = com + a * np.cos(th1) * u + a * np.sin(th1) * v
        Pc2 = com + a * np.cos(th2) * u + a * np.sin(th2) * v

        # Choose the point based on the direction of the applied moment
        vec = np.cross(Pc1 - p0, Pc1 - com)
        if np.dot(vec, M) > 0:
            p = Pc1
        else:   
            p = Pc2
    
    else:
        p = com
    
    # Compute the extended point on the VHP plane
    k = -np.dot(p-vhp_p,vhp_n)/np.dot(p0-p, vhp_n)
    pEnd = p + k * (p0 - p)


    # Plot the line segment: Xplot, Yplot, Zplot
    Xplot = np.array([p[0], pEnd[0]])
    Yplot = np.array([p[1], pEnd[1]])
    Zplot = np.array([p[2], pEnd[2]])
    return Xplot, Yplot, Zplot


def cone_surface(c: np.ndarray, 
                 mass,
                 me_mag,
                 me_phi:np.ndarray,
                 fe_mag,
                 fe_phi:np.ndarray, 
                 vhp_n:np.ndarray, 
                 p0, 
                 ns):
  
    # External moment vector
    me = np.array([me_mag*np.cos(me_phi), me_mag*np.sin(me_phi), 0]) # xy plane is horizontal
    fe = np.array([fe_mag*np.cos(fe_phi), fe_mag*np.sin(fe_phi), 0]) # xy plane is horizontal

    # R^ge, M^ge (assuming equilibrium, inertial component is zero)
    R = fe + np.array([0,0,-mass*9.81]) # g is in -z direction
    M = me

    # Compute the radius
    normR = np.linalg.norm(R)
    Gp1xR = M / normR - (np.dot(R,M))/(normR*np.dot(R,vhp_n)) * vhp_n
    a = np.linalg.norm(Gp1xR)
  
    # Unit vector along the cone axis
    # c is the COM, a is the sphere radius, p0 is the cone apex
    v = c - p0
    v /= np.linalg.norm(v)

    # Half cone angle
    dcp = np.linalg.norm(c - p0)
    alpha = np.arcsin(a/dcp)
    ta = np.tan(alpha)

    # K intersection = sphere-cone tangent
    kint = dcp - a*np.sin(alpha)

    # 2-parameter cone description
    th = np.linspace(0, 2 * np.pi, ns)
    k = np.linspace(-dcp/2, kint, ns)
    Th, K = np.meshgrid(th, k) # Create mesh grid for surface plot

    # Allocate arrays for the surface
    X_cone = np.zeros_like(Th)
    Y_cone = np.zeros_like(Th)
    Z_cone = np.zeros_like(Th)

    # Generate cone surface
    for i in range(ns):
        for j in range(ns):
            gen = np.array([np.cos(Th[i, j]), np.sin(Th[i, j]), 0])
            u = np.cross(v, gen)
            u /= np.linalg.norm(u)
            p = p0 + K[i, j] * v + K[i, j] * ta * u
            X_cone[i, j] = p[0]
            Y_cone[i, j] = p[1]
            Z_cone[i, j] = p[2]

    # Generate the circle made by sphere-cone intersection
    Xc = np.zeros_like(th)
    Yc = np.zeros_like(th)
    Zc = np.zeros_like(th)

    # K intersection = 
    kint = dcp - a*np.sin(alpha)
    for i, thi in enumerate(th):
        gen = np.array([np.cos(thi), np.sin(thi), 0])
        u = np.cross(v, gen)
        u /= np.linalg.norm(u)
        p = p0 + kint * v + kint * ta * u
        Xc[i] = p[0]
        Yc[i] = p[1]
        Zc[i] = p[2]

    return X_cone, Y_cone, Z_cone, Xc, Yc, Zc

def sphere_surface_and_more(c:np.ndarray,
                   mass,
                   me_mag,
                   me_phi:np.ndarray,
                   fe_mag,
                   fe_phi:np.ndarray,
                   vhp_n:np.ndarray,
                   ns):
    
    # External moment vector
    me = np.array([me_mag*np.cos(me_phi), me_mag*np.sin(me_phi), 0]) # xy plane is horizontal
    fe = np.array([fe_mag*np.cos(fe_phi), fe_mag*np.sin(fe_phi), 0]) # xy plane is horizontal

    # R^ge, M^ge (assuming equilibrium, inertial component is zero)
    R = fe + np.array([0,0,-mass*9.81]) # g is in -z direction
    M = me

    # Compute the radius
    normR = np.linalg.norm(R)
    Gp1xR = M / normR - (np.dot(R,M))/(normR*np.dot(R,vhp_n)) * vhp_n
    a = np.linalg.norm(Gp1xR)

    # Generate sphere surface
    phi = np.linspace(0, np.pi, ns)
    theta = np.linspace(0, 2 * np.pi, ns)
    phi, theta = np.meshgrid(phi, theta)
    X_sphere = c[0] + a * np.sin(phi) * np.cos(theta)
    Y_sphere = c[1] + a * np.sin(phi) * np.sin(theta)
    Z_sphere = c[2] + a * np.cos(phi)

    # Arrow showing direction of the Mgie
    Mhat = np.array([0, 0, 0])
    if np.linalg.norm(M) > 1e-6:
        Mhat = M / np.linalg.norm(M)
        X_me_arrow = np.array([c[0], c[0] + a*Mhat[0]])
        Y_me_arrow = np.array([c[1], c[1] + a*Mhat[1]])
        Z_me_arrow = np.array([c[2], c[2] + a*Mhat[2]])
    else:
        X_me_arrow = np.array([c[0], c[0]])
        Y_me_arrow = np.array([c[1], c[1]])
        Z_me_arrow = np.array([c[2], c[2]])

    # Arrow showing direction of Rgie
    Rhat = np.array([0, 0, 0])
    if np.linalg.norm(R) > 1e-6:
        Rhat = R / np.linalg.norm(R)
        Rhat = 0.4* c[2] * Rhat
        X_fe_arrow = np.array([c[0], c[0] + Rhat[0]])
        Y_fe_arrow = np.array([c[1], c[1] + Rhat[1]])
        Z_fe_arrow = np.array([c[2], c[2] + Rhat[2]])
    else:
        X_fe_arrow = np.array([c[0], c[0]])
        Y_fe_arrow = np.array([c[1], c[1]])
        Z_fe_arrow = np.array([c[2], c[2]])

    # Great circle normal to Mgie
    th = np.linspace(0, 2 * np.pi, ns)
    Xmc = np.zeros_like(th)
    Ymc = np.zeros_like(th)
    Zmc = np.zeros_like(th)
    for i, thi in enumerate(th):
        ang = np.arctan2(M[1], M[0])
        u = Rz(ang) @ np.array([0, np.cos(thi), np.sin(thi)])
        Xmc[i] = a*u[0] + c[0]
        Ymc[i] = a*u[1] + c[1]
        Zmc[i] = a*u[2] + c[2]

    return X_sphere, Y_sphere, Z_sphere, X_me_arrow, Y_me_arrow, Z_me_arrow, X_fe_arrow, Y_fe_arrow, Z_fe_arrow, Xmc, Ymc, Zmc

def GP1(mass, me_mag, me_phi, fe_mag, fe_phi, n_VHP, k):
    # Location of the ZMP axis w.r.t. COM

    # External moment vector
    me = np.array([me_mag*np.cos(me_phi), me_mag*np.sin(me_phi), 0]) # xy plane is horizontal
    fe = np.array([fe_mag*np.cos(fe_phi), fe_mag*np.sin(fe_phi), 0]) # xy plane is horizontal

    # R^ge, M^ge (assuming equilibrium, inertial component is zero)
    R = fe + np.array([0,0,-mass*9.81]) # g is in -z direction
    M = me

    # GP1: k is the parameter
    gp1 = np.zeros(3)

    gp1 = np.cross(R,M) / np.dot(R, R)
    gp1 -=  np.dot(R,M) * np.cross(R,n_VHP) / (np.dot(R,R) * np.dot(R,n_VHP))
    gp1 += k*R
    
    return gp1

def zmp_axis_segment(com: np.ndarray, mass, 
                     me_mag, me_phi, 
                     fe_mag, fe_phi, vhp_p, vhp_n):
    # ZMP axis:

    # Find the parameter value for a point on ZMP at 1.1 times the height of COM
    offset = GP1(mass, me_mag, me_phi, fe_mag, fe_phi, vhp_n, 0)
    fe = np.array([fe_mag*np.cos(fe_phi), fe_mag*np.sin(fe_phi), 0]) # xy plane is horizontal
    R = fe + np.array([0,0,-mass*9.81]) # g is in -z direction
    kT = (0.1 * com[2] - offset[2]) / R[2]

    # Find the parameter value for a point on VHP (ground at the moment... update later)
    kG = -(np.dot(com + offset - vhp_p, vhp_n)) / np.dot(R, vhp_n)

    # Axis is shown from this point
    gp1T = GP1(mass, me_mag, me_phi, fe_mag, fe_phi, vhp_n, kT)
    zmpT = gp1T + com

    # Axis is shown to this point
    gp1G = GP1(mass, me_mag, me_phi, fe_mag, fe_phi, vhp_n, kG)
    zmpG = gp1G + com

    # Line segment to be plotted
    Xzmp = np.array([zmpT[0], zmpG[0]])
    Yzmp = np.array([zmpT[1], zmpG[1]])
    Zzmp = np.array([zmpT[2], zmpG[2]])

    return Xzmp, Yzmp, Zzmp

def make_diagram(ax, 
                 com: np.ndarray, 
                 p0: np.ndarray, 
                 p1: np.ndarray, 
                 p2:np.ndarray,
                 p3:np.ndarray,
                 mass, 
                 me_mag, me_phi, fe_mag, fe_phi, 
                 vhp_p:np.ndarray, 
                 vhp_n:np.ndarray):

    # Clear the old lines
    if flag_trace is False:
        ax.cla()
    
    # Draw the centre of mass 
    ax.scatter(com[0], com[1], com[2], color='m', s=50, label='COM')

    # Draw the sphere surface, arrows showing me and fe
    Xs, Ys, Zs, X_m_arr, Y_m_arr, Z_m_arr, X_r_arr, Y_r_arr, Z_r_arr, Xmc, Ymc, Zmc = sphere_surface_and_more(com, 
                                                                                               mass, 
                                                                                               me_mag, me_phi, 
                                                                                               fe_mag, fe_phi, 
                                                                                               vhp_n, 
                                                                                               60)
    ax.plot_surface(Xs, Ys, Zs, alpha=0.2, color='yellow')
    
    # Draw arrows showing me and fe
    ax.quiver(X_m_arr[0], Y_m_arr[0], Z_m_arr[0],  # Start point
          X_m_arr[1] - X_m_arr[0],             # X direction
          Y_m_arr[1] - Y_m_arr[0],             # Y direction
          Z_m_arr[1] - Z_m_arr[0],             # Z direction
          color='black', 
          linewidth=3.5, 
          label=r'$M^{gie}$',
          length=1,                # Optional: adjust arrow length
          arrow_length_ratio=0.4)  # Optional: adjust arrowhead size

    ax.quiver(X_r_arr[0], Y_r_arr[0], Z_r_arr[0],  # Start point
            X_r_arr[1] - X_r_arr[0],             # X direction
            Y_r_arr[1] - Y_r_arr[0],             # Y direction
            Z_r_arr[1] - Z_r_arr[0],             # Z direction
            color='green',
            linewidth=3.5,
            label=r'$R^{gie}$',
            length=1,                # Optional: adjust arrow length
            arrow_length_ratio=0.4)  # Optional: adjust arrowhead size

    # Draw great circle normal to Mgie
    ax.plot(Xmc, Ymc, Zmc, color='black')

    # Draw the zmp axis
    Xzmp, Yzmp, Zzmp = zmp_axis_segment(com, mass, 
                                        me_mag, me_phi, 
                                        fe_mag, fe_phi, 
                                        vhp_p, vhp_n)
    ax.plot(Xzmp, Yzmp, Zzmp, color='b', label='ZMP axis', linewidth=2.5)
    # Draw ZMP on VHP
    ax.scatter(Xzmp[1], Yzmp[1], Zzmp[1], color='blue', s=50, marker='s', label='ZMP on VHP')
    # Draw ZMP in the feet plane
    Xzf, Yzf, Zzf = zmp_axis_segment(com, mass,
                                        me_mag, me_phi,
                                        fe_mag, fe_phi,
                                        p0, vhp_n)
    ax.scatter(Xzf[1], Yzf[1], Zzf[1], color='blue', s=50, marker='s')

    # Draw a cone surface from point P0
    X_cone, Y_cone, Z_cone, Xcc, Ycc, Zcc = cone_surface(com, mass,
                                             me_mag, me_phi, 
                                             fe_mag, fe_phi, 
                                             vhp_n, p0, 50)

    ax.plot_surface(X_cone, Y_cone, Z_cone, color='orange', alpha=0.2)
    ax.plot(Xcc, Ycc, Zcc, color='orange', linewidth=3)

    
    # Draw the feet points p0---p3
    pmat = np.array([p0, p1, p2, p3, p0])
    ax.plot(pmat[:,0], pmat[:,1], pmat[:,2], color='black', label='Feet points', marker='o', linewidth=0.5, linestyle='dotted')
    
    # Draw the old projection on VHP
    Xold0, Yold0, Zold0 = old_proj(com, p0, vhp_n, vhp_p)
    Xold1, Yold1, Zold1 = old_proj(com, p1, vhp_n, vhp_p)
    Xold2, Yold2, Zold2 = old_proj(com, p2, vhp_n, vhp_p)
    Xold3, Yold3, Zold3 = old_proj(com, p3, vhp_n, vhp_p)
    ax.plot(Xold0, Yold0, Zold0, color='red', linewidth=1, linestyle='dotted')
    ax.plot(Xold1, Yold1, Zold1, color='red', linewidth=1, linestyle='dotted')
    ax.plot(Xold2, Yold2, Zold2, color='red', linewidth=1, linestyle='dotted')
    ax.plot(Xold3, Yold3, Zold3, color='red', linewidth=1, linestyle='dotted')

    # Plot the old support polygon on VHP
    XYZoldmat = np.array([[Xold0[1], Yold0[1], Zold0[1]], 
                          [Xold1[1], Yold1[1], Zold1[1]], 
                          [Xold2[1], Yold2[1], Zold2[1]], 
                          [Xold3[1], Yold3[1], Zold3[1]], 
                          [Xold0[1], Yold0[1], Zold0[1]]])
    ax.plot(XYZoldmat[:,0], XYZoldmat[:,1], XYZoldmat[:,2], color='red', linewidth=1, linestyle='dotted', marker='o', label='old method')

    # Draw the new projection on VHP
    Xnew0, Ynew0, Znew0 = new_proj(com, mass, me_mag, me_phi, p0, vhp_p, vhp_n)
    Xnew1, Ynew1, Znew1 = new_proj(com, mass, me_mag, me_phi, p1, vhp_p, vhp_n)
    Xnew2, Ynew2, Znew2 = new_proj(com, mass, me_mag, me_phi, p2, vhp_p, vhp_n)
    Xnew3, Ynew3, Znew3 = new_proj(com, mass, me_mag, me_phi, p3, vhp_p, vhp_n)
    ax.plot(Xnew0, Ynew0, Znew0, color='green', linewidth=1.5, linestyle='dotted')
    ax.plot(Xnew1, Ynew1, Znew1, color='green', linewidth=1.5, linestyle='dotted')
    ax.plot(Xnew2, Ynew2, Znew2, color='green', linewidth=1.5, linestyle='dotted')
    ax.plot(Xnew3, Ynew3, Znew3, color='green', linewidth=1.5, linestyle='dotted')

    # Plot the new support polygon on VHP
    XYZnewmat = np.array([[Xnew0[1], Ynew0[1], Znew0[1]], 
                          [Xnew1[1], Ynew1[1], Znew1[1]], 
                          [Xnew2[1], Ynew2[1], Znew2[1]], 
                          [Xnew3[1], Ynew3[1], Znew3[1]], 
                          [Xnew0[1], Ynew0[1], Znew0[1]]])
    ax.plot(XYZnewmat[:,0], XYZnewmat[:,1], XYZnewmat[:,2], color='green', linewidth=1.5, linestyle='dotted', marker='o', label='new method')                          
    
    # Draw a frame at origin
    origin = [0, 0, 0]
    x_axis = [0.01, 0, 0]
    y_axis = [0, 0.01, 0]
    z_axis = [0, 0, 0.01]
    # Plot the quiver (arrow) plot
    ax.quiver(origin[0], origin[1], origin[2],
              x_axis[0], x_axis[1], x_axis[2],
              color='red')
    ax.quiver(origin[0], origin[1], origin[2],
              y_axis[0], y_axis[1], y_axis[2],
              color='green')
    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0], z_axis[1], z_axis[2],
              color='blue')


    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_xlim([-0.06, 0.06])
    ax.set_ylim([-0.06, 0.06])
    ax.set_zlim([0, 0.16])

    # Show the figure
    plt.draw()


# Robot key points location
mass = 1
com = np.array([0.015, 0.015, 0.1])

# Define the external point
p0 = np.array([0.03, 0.03, 0.03])
p1 = np.array([0.03, -0.03, 0.03])
p2 = np.array([0, -0.03, 0.03])
p3 = np.array([0, 0.03, 0.03])

# VHP
nVHP = np.array([0, 0, 1])
pVHP = np.array([0, 0, 0.])

# number of samples for plotting
ns = 50

# Make a new 2D figure
fig = plt.figure(figsize=(10, 10))
# Add an axis to the figure (20% from left, 30% from bottom, 60% width, 60% height)
ax = fig.add_axes([0.3, 0.1, 0.6, 0.6], projection='3d')

# flag for traceON/OFF
flag_trace = False

# Add axis labes x y and z
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Axis limits
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(-0.1, 0.1)
ax.set_zlim(0, 0.2)


# Initial plot
make_diagram(ax, com, p0, p1, p2, p3, mass, 0, 0, 0, 0, pVHP, nVHP)

# Add sliders
ax_me_mag = plt.axes((0.05, 0.1, 0.02, 0.8))
me_mag_slider = Slider(ax=ax_me_mag,
                   label='me',
                   valmin=-1,
                   valmax=+1,
                   valinit=0,
                   orientation='vertical')

ax_me_dir = plt.axes((0.1, 0.1, 0.02, 0.8))
me_dir_slider = Slider(ax=ax_me_dir,
                   label='me_dir',
                   valmin=-np.pi,
                   valmax=+np.pi,
                   valinit=0,
                   orientation='vertical')

ax_fe_mag = plt.axes((0.15, 0.1, 0.02, 0.8))
fe_mag_slider = Slider(ax=ax_fe_mag,
                   label='fe',
                   valmin=-1.*mass*9.81,
                   valmax=+1.*mass*9.81,
                   valinit=0,
                   orientation='vertical')

ax_fe_dir = plt.axes((0.2, 0.1, 0.02, 0.8))
fe_dir_slider = Slider(ax=ax_fe_dir,
                   label='fe_dir',
                   valmin=-np.pi,
                   valmax=+np.pi,
                   valinit=0,
                   orientation='vertical')

# Button
ax_reset_button = plt.axes((0.8, 0.8, 0.1, 0.05))
reset_button = Button(ax_reset_button, 'Reset')

ax_trace_button = plt.axes((0.8, 0.7, 0.1, 0.05))
trace_button = Button(ax_trace_button, 'Trace')

def update(val):
    make_diagram(ax, com, p0, p1, p2, p3, mass, 
                 me_mag_slider.val, me_dir_slider.val, 
                 fe_mag_slider.val, fe_dir_slider.val, 
                 pVHP, nVHP)
    fig.canvas.draw_idle()

# Set the callback function
me_mag_slider.on_changed(update)
me_dir_slider.on_changed(update)
fe_mag_slider.on_changed(update)
fe_dir_slider.on_changed(update)

reset_button.on_clicked(lambda x: [me_mag_slider.reset(), 
                                   me_dir_slider.reset(), 
                                   fe_mag_slider.reset(), 
                                   fe_dir_slider.reset(), 
                                   make_diagram(ax, com, p0, p1, p2, p3, mass, 
                                                me_mag_slider.val, me_dir_slider.val, 
                                                fe_mag_slider.val, fe_dir_slider.val, 
                                                pVHP, nVHP)])
# Toggle the flag_trace state on click
trace_button.on_clicked(lambda x: [update(x), [globals().update({'flag_trace': not flag_trace})]])

plt.show()
