"""Same FP32 collision arithmetic; reuse per-trajectory scratch buffers."""
import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads
F = np.float32

@njit(nogil=True, fastmath=False, cache=False)
def configuration(q, parent, child, coordinate, origin, translation, links, local, radii, pairs, types, centers, rotations, dimensions, rotation, position, spheres, relative):
    rotation.fill(F(0))
    position.fill(F(0))
    for i in range(3):
        rotation[0, i, i] = F(1)
    for j in range(len(parent)):
        p, c = (parent[j], child[j])
        for a in range(3):
            position[c, a] = position[p, a]
            for b in range(3):
                position[c, a] += rotation[p, a, b] * translation[j, b]
        for a in range(3):
            for b in range(3):
                relative[a, b] = origin[j, a, b]
        if coordinate[j] >= 0:
            cosine, sine = (F(np.cos(np.float64(q[coordinate[j]]))), F(np.sin(np.float64(q[coordinate[j]]))))
            for a in range(3):
                x, y = (origin[j, a, 0], origin[j, a, 1])
                relative[a, 0] = cosine * x + sine * y
                relative[a, 1] = -sine * x + cosine * y
        for a in range(3):
            for b in range(3):
                for k in range(3):
                    rotation[c, a, b] += rotation[p, a, k] * relative[k, b]
    for s in range(len(links)):
        link = links[s]
        for a in range(3):
            spheres[s, a] = position[link, a]
            for b in range(3):
                spheres[s, a] += rotation[link, a, b] * local[s, b]
    world, self_distance = (F(1000000.0), F(1000000.0))
    for s in range(len(spheres)):
        for obj in range(len(types)):
            x, y, z = (spheres[s, 0] - centers[obj, 0], spheres[s, 1] - centers[obj, 1], spheres[s, 2] - centers[obj, 2])
            px = rotations[obj, 0, 0] * x + rotations[obj, 1, 0] * y + rotations[obj, 2, 0] * z
            py = rotations[obj, 0, 1] * x + rotations[obj, 1, 1] * y + rotations[obj, 2, 1] * z
            pz = rotations[obj, 0, 2] * x + rotations[obj, 1, 2] * y + rotations[obj, 2, 2] * z
            if types[obj] == 0:
                dx, dy, dz = (abs(px) - dimensions[obj, 0], abs(py) - dimensions[obj, 1], abs(pz) - dimensions[obj, 2])
                distance = F(np.power(np.float64(max(dx, F(0)) ** 2 + max(dy, F(0)) ** 2 + max(dz, F(0)) ** 2), np.float64(F(0.5)))) + min(max(dx, dy, dz), F(0))
            elif types[obj] == 1:
                radial = F(np.power(np.float64(px ** 2 + py ** 2), np.float64(F(0.5)))) - dimensions[obj, 1]
                axial = abs(pz) - dimensions[obj, 0]
                distance = F(np.power(np.float64(max(radial, F(0)) ** 2 + max(axial, F(0)) ** 2), np.float64(F(0.5)))) + min(max(radial, axial), F(0))
            else:
                distance = F(np.power(np.float64(px ** 2 + py ** 2 + pz ** 2), np.float64(F(0.5)))) - dimensions[obj, 0]
            world = min(world, distance - radii[s])
    for i in range(len(pairs)):
        a, b = (pairs[i, 0], pairs[i, 1])
        dx, dy, dz = (spheres[a, 0] - spheres[b, 0], spheres[a, 1] - spheres[b, 1], spheres[a, 2] - spheres[b, 2])
        self_distance = min(self_distance, F(np.power(np.float64(dx ** 2 + dy ** 2 + dz ** 2), np.float64(F(0.5)))) - radii[a] - radii[b])
    return (world, self_distance)

@njit(nogil=True, parallel=True, fastmath=False, cache=False)
def _kernel_batch(q, parent, child, coordinate, origin, translation, links, local, radii, pairs, types, centers, rotations, dimensions):
    result = np.empty((len(q), q.shape[1], 2), np.float32)
    total = len(q) * q.shape[1]
    if total == 0:
        return result
    blocks = min(total, get_num_threads())
    for block in prange(blocks):
        rotation = np.empty((13, 3, 3), np.float32)
        position = np.empty((13, 3), np.float32)
        spheres = np.empty((len(links), 3), np.float32)
        relative = np.empty((3, 3), np.float32)
        start = total * block // blocks
        stop = total * (block + 1) // blocks
        for index in range(start, stop):
            row = index // q.shape[1]
            sample = index - row * q.shape[1]
            a, b = configuration(q[row, sample], parent, child, coordinate, origin, translation, links, local, radii, pairs, types, centers, rotations, dimensions, rotation, position, spheres, relative)
            result[row, sample, 0] = a
            result[row, sample, 1] = b
    return result

def batch(q, parent, child, coordinate, origin, translation, links, local, radii, pairs, types, centers, rotations, dimensions):
    workers = get_num_threads()
    serial_small = workers > 1 and len(q) * q.shape[1] < 64
    if serial_small:
        set_num_threads(1)
    try:
        return _kernel_batch(q, parent, child, coordinate, origin, translation, links, local, radii, pairs, types, centers, rotations, dimensions)
    finally:
        if serial_small:
            set_num_threads(workers)
