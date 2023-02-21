import numpy as np
from typing import Tuple


def rounded_square(
    resolution: float,
    phys_size: Tuple[float, float],
    declared_mls: float,
    angle: float = 0,
    center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    '''
    phys_size: array or list with 2 elements that describe the physical size of the design pattern
    resolution: number of points per unit length
    declared_mls: declared minimum length scale
    center: array or list with 2 elements that describe the center position of the square
    angle: angle of rotation, in degree
    '''
    phys_size = np.array(phys_size)
    angle = np.radians(angle)

    grid_size = phys_size - 1 / resolution
    n = np.round(phys_size * resolution).astype(int)

    x_coord = np.linspace(-grid_size[0] / 2, grid_size[0] / 2, n[0])
    y_coord = np.linspace(-grid_size[1] / 2, grid_size[1] / 2, n[1])
    xv, yv = np.meshgrid(x_coord, y_coord, sparse=True, indexing='ij')

    side, diameter = 2 * declared_mls, declared_mls
    rect_vert = np.where(abs(np.sin(angle)*(xv-center[0])-np.cos(angle)*(yv-center[1]))<=(side-diameter)/2,1,0)*\
                np.where(abs(np.cos(angle)*(xv-center[0])+np.sin(angle)*(yv-center[1]))<=side/2,1,0)
    rect_hori = np.where(abs(np.sin(angle)*(xv-center[0])-np.cos(angle)*(yv-center[1]))<=side/2,1,0)*\
                np.where(abs(np.cos(angle)*(xv-center[0])+np.sin(angle)*(yv-center[1]))<=(side-diameter)/2,1,0)

    disc_centers = np.array([[
        side - diameter, diameter - side, diameter - side, side - diameter
    ], [side - diameter, side - diameter, diameter - side, diameter - side]
                             ]) / 2
    disc_centers_x = disc_centers[0, :] * np.cos(angle) - disc_centers[
        1, :] * np.sin(angle) + center[0]
    disc_centers_y = disc_centers[0, :] * np.sin(angle) + disc_centers[
        1, :] * np.cos(angle) + center[1]
    disc_centers = np.vstack((disc_centers_x, disc_centers_y))

    disc0 = np.where(
        abs(xv - disc_centers[0, 0])**2 + abs(yv - disc_centers[1, 0])**2 <=
        diameter**2 / 4, 1, 0)
    disc1 = np.where(
        abs(xv - disc_centers[0, 1])**2 + abs(yv - disc_centers[1, 1])**2 <=
        diameter**2 / 4, 1, 0)
    disc2 = np.where(
        abs(xv - disc_centers[0, 2])**2 + abs(yv - disc_centers[1, 2])**2 <=
        diameter**2 / 4, 1, 0)
    disc3 = np.where(
        abs(xv - disc_centers[0, 3])**2 + abs(yv - disc_centers[1, 3])**2 <=
        diameter**2 / 4, 1, 0)

    return rect_vert + rect_hori + disc0 + disc1 + disc2 + disc3 > 0.1


def disc(resolution: float,
         phys_size: Tuple[float, float],
         diameter: float,
         center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    '''
    phys_size: array or list with 2 elements that describe the physical size of the design pattern
    resolution: number of points per unit length
    declared_mls: declared minimum length scale
    center: array or list with 2 elements that describe the center position of the square
    angle: angle of rotation, in degree
    '''
    phys_size = np.array(phys_size)
    grid_size = phys_size - 1 / resolution
    n = np.round(phys_size * resolution).astype(int)

    x_coord = np.linspace(-grid_size[0] / 2, grid_size[0] / 2, n[0])
    y_coord = np.linspace(-grid_size[1] / 2, grid_size[1] / 2, n[1])
    xv, yv = np.meshgrid(x_coord, y_coord, sparse=True, indexing='ij')
    disc = np.where(
        abs(xv - center[0])**2 + abs(yv - center[1])**2 <= diameter**2 / 4, 1,
        0)

    return disc > 0.1
