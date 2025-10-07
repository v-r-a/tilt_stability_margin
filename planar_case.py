# Copyright (c) 2025 Vyankatesh Ashtekar
# All rights reserved.
# This code is provided for academic review only.
# No permission is granted for reuse, modification, or redistribution.

# Note that the limits on forces and moments are not enforced in the demo utilities and the ZMP axis may be go outside the foot supports. Values of forces/moments corresponding to such cases should be assumed as infeasible.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# Drawings
def circ_coord(cen:np.ndarray, mass, me, fe, ns):
    # radius of the circle in static case
    rad = np.abs(me) / np.sqrt((mass*9.81)**2 + fe**2)

    # Coordinates
    th = np.linspace(0, 2*np.pi, ns)
    x = np.zeros_like(th)
    y = np.zeros_like(th)
    for i, thi in enumerate(th):
        x[i] = cen[0] + rad*np.cos(thi)
        y[i] = cen[1] + rad*np.sin(thi)
    return x, y

def tgt_pt_circ(pt:np.ndarray, cen:np.ndarray, me, fe):
    # tangent from pt to circle with cen at com and rad based on me
    # choice amongst the two tgts will be done by the sign of me

    # radius of the circle in static case
    rad = np.abs(me) / np.sqrt((mass*9.81)**2 + fe**2)

    # Solve for the two touch points analytically
    # Shift the origin to the centre of the circle (COM)
    x0 = pt[0] - cen[0] # x coord
    y0 = pt[1] - cen[1] # y-coord

    den = x0**2 + y0**2
    cx0y0 = den - rad**2

    # touch point 1
    x1 = x0*(rad**2) - y0*rad*np.sqrt(cx0y0)
    x1 /= den

    y1 = rad**2 - x0*x1
    y1 /= y0

    # touch point 2
    x2 = x0*(rad**2) + y0*rad*np.sqrt(cx0y0)
    x2 /= den

    y2 = rad**2 - x0*x2
    y2 /= y0

    # Shift back the frame to origin
    x0 += cen[0]
    y0 += cen[1]
    x1 += cen[0]
    y1 += cen[1]
    x2 += cen[0]
    y2 += cen[1]
    
    # Choose the correct tangent (pt-p1 or pt-p2) based on the moment
    v1 = np.cross(np.array([x1,y1,0])-cen, np.array([x0,y0,0])-cen)
    if np.dot(v1,[0,0,me]) > 0:
        # Shift back the frame to origin
        return [x0 , x1], [y0 , y1]
    else:
        # Shift back the frame to origin
        return [x0, x2], [y0, y2]

def proj_vhp(pt:np.ndarray, cen:np.ndarray, me, fe, vhp_m, vhp_c):
    # x0t = [pt[0],xtgt]
    # y0t = [pt[1], ytgt]
    x0t, y0t = tgt_pt_circ(pt, cen, me, fe)

    x0 = pt[0]
    y0 = pt[1]
    x1 = x0t[1]
    y1 = y0t[1]

    # intersection points
    xint = 0
    yint = 0

    # Find the intersection of lines P0--Ptgt and the VHP
    # The VHP is a line in 2D defined by: y = vhp_m x + vhp_c
    if np.abs(x1-x0) < 1e-6:
        xint = x0
        yint = vhp_m * xint + vhp_c
    else:
        m = (y1-y0)/(x1-x0)
        xint = ((y0-m*x0) - vhp_c)/(vhp_m - m)
        yint = vhp_m * xint + vhp_c

    return [x1,xint], [y1,yint]

def zmp_axis(com:np.ndarray, mass, me, fe, vhp_m, vhp_c):
    # ZMP axis: Ax + By + C =0

    A = mass * 9.81
    B = fe
    C = me - com[0] * mass * 9.81  - com[1] * fe

    # intersection of the ZMP axis with the VHP
    xint=0
    yint=0
    if np.abs(B) < 1e-6:
        xint = -C/A
        yint = vhp_m * xint + vhp_c
    else:
        xint = -C - vhp_c*B
        xint /= (A + B*vhp_m)
        yint = vhp_m * xint + vhp_c

    # Extend the display of ZMP axis 1.x times the height of COM
    ytop = 1.1*com[1]
    xtop = -C - B*ytop
    xtop /= A

    return [xint, xtop], [yint, ytop]

def make_diagram(ax, com:np.ndarray, p0:np.ndarray, p1:np.ndarray, mass, me, fe, vhp_m, vhp_c):
    # Draw the points p0 and p1
    ax.plot(p0[0], p0[1], 'ro', label='feet')
    ax.plot(p1[0], p1[1], 'ro')

    # Draw the circle and its center
    cen = com
    x, y = circ_coord(cen, mass, me, fe, 50)
    ax.plot(x, y, linestyle='dotted' ,color='magenta')
    ax.plot(cen[0], cen[1], 'mo', markersize=4, label='COM')

    # Draw the projectors
    x1, y1 = proj_vhp(p0, cen, me, fe, vhp_m, vhp_c)
    # x1, y1 = tgt_pt_circ(p0, cen, me, fe)
    ax.plot(x1, y1, 'g', label='new method') 
    x2, y2 = proj_vhp(p1, cen, me, fe, vhp_m, vhp_c)
    # x2, y2 = tgt_pt_circ(p1, cen, me, fe)
    ax.plot(x2, y2, 'g')

    # Draw the old projectors
    x1, y1 = proj_vhp(p0, cen, 0, fe, vhp_m, vhp_c)
    # x1, y1 = tgt_pt_circ(p0, cen, me, fe)
    ax.plot(x1, y1, 'r', linestyle='--',linewidth=0.5, label='old method')
    x2, y2 = proj_vhp(p1,cen,0,fe,vhp_m,vhp_c)
    # x2, y2 = tgt_pt_circ(p1, cen, me, fe)
    ax.plot(x2, y2, 'r', linestyle='--', linewidth=0.5)

    # Draw the ZMP axis
    xz, yz = zmp_axis(com, mass, me, fe, vhp_m, vhp_c)
    ax.plot(xz, yz, 'b',label='ZMP axis')
    ax.plot(xz[0],yz[0], 'b', marker='s', label='ZMP on VHP')

    # Draw the force R^gie
    sc = 0.02 * com[1]
    ax.arrow(com[0], com[1],
             sc * fe, -sc * mass * 9.81,
              color='gray',
              linewidth=1)

    # Draw the VHP
    xv1 = np.min([p0[0], p1[0], cen[0], x1[0], x2[0], xz[0]]) - 0.05
    xv2 = np.max([p0[0], p1[0], cen[0], x1[0], x2[0], xz[0]]) + 0.05
    yv1 = vhp_m * xv1 + vhp_c
    yv2 = vhp_m * xv2 + vhp_c
    ax.plot([xv1, xv2], [yv1, yv2],'k', linewidth=1, label='VHP')

    # Show the legend outside and above the plot
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

    # Set the aspect ratio to 1
    ax.set_aspect('equal')
    # Set the limits
    xmin = np.min([p0[0], p1[0], cen[0], x1[0], x2[0], xz[0]])
    xmax = np.max([p0[0], p1[0], cen[0], x1[0], x2[0], xz[0]])
    ax.set_xlim(xmin-0.05, xmax+0.05)
    ymin = np.min([p0[1], p1[1], cen[1], y1[0], y2[0], yz[0], yv1, yv2]) - 0.02
    ax.set_ylim(ymin, 1.3*com[1])

    # Show the figure
    plt.draw()

# Robot key points location
mass = 1
com = np.array([0.015, 0.06, 0.])
p0 = np.array([0.0, 0.01, 0.])
p1 = np.array([0.03, 0.02, 0.])

# VHP
mVHP = 0
cVHP = 0

# Make a new 2D figure
fig = plt.figure(figsize=(10, 10))

# Add an axis to the figure (20% from left, 30% from bottom, 60% width, 60% height)
ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

# Initial plot
make_diagram(ax, com, p0, p1, mass, 0, 0, mVHP, cVHP)

# Add sliders
ax_me = plt.axes((0.05, 0.1, 0.02, 0.8))
me_slider = Slider(ax=ax_me,
                   label='me',
                   valmin=-1. * (mass * 9.81 * np.abs(com[0]-p0[0])),
                   valmax=+1. * (mass * 9.81 * np.abs(p1[0]-com[0])),
                   valinit=0,
                   orientation='vertical')

ax_fe = plt.axes((0.1, 0.1, 0.02, 0.8))
fe_slider = Slider(ax=ax_fe,
                   label='fe',
                   valmin=-0.3 * (mass * 9.81),
                   valmax=+0.3 * (mass * 9.81),
                   valinit=0,
                   orientation='vertical')

ax_cVHP = plt.axes((0.35, 0.15, 0.45, 0.05))
cVHP_slider = Slider(ax=ax_cVHP,
                     label='VHP offset',
                     valmin=-0.1,
                     valmax=+0.01,
                     valinit=0,
                     orientation='horizontal')

# Button
ax_reset_button = plt.axes((0.8, 0.5, 0.1, 0.05))
reset_button = Button(ax_reset_button, 'Reset')

def update(val):
    ax.clear()
    make_diagram(ax, com, p0, p1, mass, me_slider.val, fe_slider.val, 0., cVHP_slider.val)
    fig.canvas.draw_idle()

# Set the callback function
me_slider.on_changed(update)
fe_slider.on_changed(update)
cVHP_slider.on_changed(update)

reset_button.on_clicked(lambda x: [me_slider.reset(), fe_slider.reset(), cVHP_slider.reset()])

plt.show()
