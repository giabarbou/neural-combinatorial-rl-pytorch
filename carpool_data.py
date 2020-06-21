import csv
import numpy as np
import torch


# save coordinates to file readable by Google My Maps
def save_coordinates(fname, coords, ids=None, extras=None):
    if extras is None:
        extras = ['' for i in range(len(coords))]

    if ids is None:
        ids = [i for i in range(len(coords))]

    with open(fname, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['latitude', 'longitude', 'id', 'extras'])
        for i in range(len(coords)):
            writer.writerow([str(coords[i][0]), str(coords[i][1]), str(ids[i]), str(extras[i])])


# load map coordinates from file
def load_coordinates(fname):
    with open(fname, 'r', encoding='utf8') as file:
        reader = csv.DictReader(file, delimiter=',')
        coords = [[float(row['latitude']), float(row['longitude'])] for row in reader]

    return coords


# load a distance matrix from file
def load_matrix(fname):
    dist = []
    with open(fname, 'r', encoding='utf8') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            dist.append([float(d) for d in row])

    return dist


class CarpoolDataManager:
    """
        A class that manages the data needed for carpooling purposes.
        It contains a set of coordinates and their distances and can sample from them,
        normalize the data and make submatrices of the initial data if needed.
    """
    def set_data(self):
        # initializes the class fields

        self.d_min = np.min(self.dist[self.dist > 0])
        self.d_max = np.max(self.dist)

        self.lat_max = np.max(self.coords[:, 0])
        self.lat_min = np.min(self.coords[:, 0])

        self.long_max = np.max(self.coords[:, 1])
        self.long_min = np.min(self.coords[:, 1])

        coords_n = self.coords.copy()
        coords_n[:, 0] = (coords_n[:, 0] - self.lat_min) / \
                         (self.lat_max - self.lat_min)
        coords_n[:, 1] = (coords_n[:, 1] - self.long_min) / \
                         (self.long_max - self.long_min)

        self.ids = {}
        for i in range(len(self.coords)):
            self.ids[(self.coords[i][0], self.coords[i][1])] = i
            self.ids[(coords_n[i][0], coords_n[i][1])] = i

    def __init__(self, dist_fname, coords_fname):
        """
        Args:
            dist_fname (string): distance matrix file that specifies the distances between map coordinates
            coords_fname (string): file with map coordinates
        """
        self.dist_fname = dist_fname
        self.coords_fname = coords_fname

        self.dist = load_matrix(self.dist_fname)
        self.coords = load_coordinates(self.coords_fname)

        self.coords = np.asarray(self.coords, dtype=np.float32)
        self.dist = np.asarray(self.dist, dtype=np.float32)

        self.set_data()

    def reset_data(self):
        # resets the class fields

        self.dist = load_matrix(self.dist_fname)
        self.coords = load_coordinates(self.coords_fname)

        self.coords = np.asarray(self.coords, dtype=np.float32)
        self.dist = np.asarray(self.dist, dtype=np.float32)

        self.set_data()

    def filter_data(self, max_range, idx_goal):
        # filter coordinates so that the class contains only those that are within a certain range from a specific
        # point indexed by idx_goal in the distance matrix

        self.coords = np.asarray(self.coords, dtype=np.float32)
        self.dist = np.asarray(self.dist, dtype=np.float32)

        if max_range is None or idx_goal is None:
            return

        idxs = np.argwhere(self.dist[:, idx_goal] <= max_range)[:, 0]

        dist_new = np.zeros([len(idxs), len(idxs)], dtype=self.dist.dtype)
        k = 0
        for i in idxs:
            dist_new[k, :] = self.dist[i, idxs]
            k += 1

        self.dist = dist_new
        self.coords = self.coords[idxs].copy()

        self.set_data()

    def pts2ids(self, points):
        ids = [self.ids[p[0], p[1]] for p in points]
        return ids

    def ids2pts(self, ids):
        return self.coords[ids].copy()

    def distances_ids(self, ids):
        # return the distances of some points indexed by ids

        n = len(ids)
        sample_dist = np.zeros([n, n])
        sample_coords = self.coords[ids, :].copy()
        k = 0
        for i in ids:
            sample_dist[k, :] = self.dist[i, ids]
            k += 1
        return sample_dist, sample_coords

    def distances_pts(self, points):
        n = len(points)
        ids = self.pts2ids(points)
        dist = np.zeros([n, n])
        k = 0
        for i in ids:
            dist[k, :] = self.dist[i, ids]
            k += 1
        return dist

    def sample_data(self, num_points, idx_goal=None, seed=None, calculate_dist=True):
        # sample some random carpool data

        if seed is not None:
            np.random.seed(seed)

        ids = np.arange(len(self.coords), dtype=np.uint32)

        np.random.shuffle(ids)
        if idx_goal is not None:
            idx = (ids == idx_goal)
            ids[idx], ids[0] = ids[0], ids[idx]

        r_ids = ids[:num_points]

        sample_coords = self.ids2pts(r_ids)

        sample_dist = None
        if calculate_dist:
            sample_dist = np.zeros([num_points, num_points], dtype=self.dist.dtype)
            k = 0
            for i in r_ids:
                sample_dist[k, :] = self.dist[i, r_ids]
                k += 1

        return sample_coords, r_ids, sample_dist

    def normalize(self, points):
        points_n = np.asarray(points, dtype=np.float32).copy()
        if points_n.ndim == 1:
            points_n[0] = (points_n[0] - self.lat_min) / (self.lat_max - self.lat_min)
            points_n[1] = (points_n[1] - self.long_min) / (self.long_max - self.long_min)
        else:
            points_n[:, 0] = (points_n[:, 0] - self.lat_min) / (self.lat_max - self.lat_min)
            points_n[:, 1] = (points_n[:, 1] - self.long_min) / (self.long_max - self.long_min)

        return points_n

    def denormalize(self, points):
        _points = np.asarray(points, dtype=np.float32).copy()
        if _points.ndim == 1:
            _points[0] = _points[0] * (self.lat_max - self.lat_min) + self.lat_min
            _points[1] = _points[1] * (self.long_max - self.long_min) + self.long_min
        else:
            _points[:, 0] = _points[:, 0] * (self.lat_max - self.lat_min) + self.lat_min
            _points[:, 1] = _points[:, 1] * (self.long_max - self.long_min) + self.long_min

        return _points

    def tensor_distance_2_pts(self, p1, p2):
        id1 = self.ids[p1[0].item(), p1[1].item()]
        id2 = self.ids[p2[0].item(), p2[1].item()]

        d = self.dist[id1][id2]

        return d

    def tensor_pts2ids(self, points):
        ids = [self.ids[p[0].item(), p[1].item()] for p in points]
        return ids

    def tensor_distances(self, t1, t2):
        t_size = len(t1)
        dist = [self.tensor_distance_2_pts(t1[i], t2[i]) for i in range(t_size)]
        dist = torch.tensor(dist, dtype=torch.float32)

        return dist

    def tensor_distances_ids(self, ids1, ids2):
        t_size = len(ids1)
        dist = [self.dist[ids1[i], ids2[i]] for i in range(t_size)]
        dist = torch.tensor(dist, dtype=torch.float32)

        return dist


if __name__ == '__main__':
    coords_fname = 'map_data/carpool_map_coordinates.csv'
    dist_fname = 'map_data/distance_matrix.csv'

    mgr = CarpoolDataManager(dist_fname, coords_fname)
    mgr.filter_data(3000, idx_goal=0)

    coords, ids, dist = mgr.sample_data(20, idx_goal=0)

    mgr.reset_data()