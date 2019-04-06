import math
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import chain
from sklearn.manifold import Isomap, MDS, LocallyLinearEmbedding


def spiral(radius, step, resolution=.1, angle=0.0, start=0.0, spread=0.1):
    dist = start + 0.0
    x = []
    y = []
    while dist * math.hypot(math.cos(angle), math.sin(angle)) < radius:
        x.append(dist * math.cos(angle) + random.uniform(-1, 1) * spread)
        y.append(dist * math.sin(angle) + random.uniform(-1, 1) * spread)
        dist += step
        angle += resolution
    plt.clf()
    plt.scatter(x, y)
    title = "spiral={0:.2f}.png".format(resolution)
    plt.title(title)
    plt.savefig(title)
    return [x, y]


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def half_moon_spiral(resolution=0.1, radius=1, spiral_density=.1):
    angle = 0.0;
    x = []
    y = []
    z = []
    while angle <= math.pi:
        pos_x = math.cos(angle) * radius
        pos_z = math.sin(angle) * radius
        sp = spiral(10, step=0.05, resolution=spiral_density, spread=.5)
        for i in range(len(sp[0])):
            x_pos = sp[0][i] + pos_x
            y_pos = sp[1][i]
            z_pos = pos_z
            x.append(x_pos)
            y.append(y_pos)
            z.append(z_pos)
        angle += resolution

    return [x, y, z]


def draw_mds(matrix, spiral_density, layer_distance):
    embedding = MDS(n_components=2, max_iter=100)
    mds = embedding.fit_transform(matrix)
    plt.clf()
    plt.scatter(mds[:, 0], mds[:, 1])
    title = "mds_spiral_density={0:.2f}_layer_distance={1:.2f}.png".format(spiral_density, layer_distance)
    plt.title(title)
    plt.savefig(title)


def draw_iso_map(matrix, spiral_density, layer_distance, k):
    embedding = Isomap(n_components=2, n_neighbors=k)
    iso_map = embedding.fit_transform(matrix)
    plt.clf()
    plt.scatter(iso_map[:, 0], iso_map[:, 1])
    title = "iso_map_spiral_density={0:.2f}_layer_distance={1:.2f}_k={2:.2f}.png".format(spiral_density, layer_distance,
                                                                                         k)
    plt.title = title
    plt.savefig(title)


def draw_lle(matrix, spiral_density, layer_distance, k):
    embedding = LocallyLinearEmbedding(n_components=2)
    lle = embedding.fit_transform(matrix)
    plt.clf()
    plt.scatter(lle[:, 0], lle[:, 1])
    title = "lle_spiral_density={0:.2f}_layer_distance={1:.2f}_k={2:.2f}.png".format(spiral_density, layer_distance, k)
    plt.title(title)
    plt.savefig(title)


for density in [.05, .1, .12]:
    for layer_distance in [.1, .15, .2]:
        hm_spiral = half_moon_spiral(radius=15, resolution=layer_distance, spiral_density=density)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(hm_spiral[0], hm_spiral[1], hm_spiral[2])
        title = "half_moon_spiral_density={0:.2f}_layer_distance={1:.2f}.png".format(density, layer_distance)
        plt.title(title)
        plt.savefig(title)
        matrix = np.column_stack((hm_spiral[0], hm_spiral[1], hm_spiral[2]))
        draw_mds(matrix, density, layer_distance)
        for k in [3, 5, 7]:
            draw_iso_map(matrix, density, layer_distance, k)
            draw_lle(matrix, density, layer_distance, k)
