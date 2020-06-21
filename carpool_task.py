import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from tqdm import trange, tqdm
import os
from carpool_data import CarpoolDataManager
import numpy as np

# don't change these indices
IDX_THR = 0
IDX_GOAL = 1
IDX_START = 2

# carpool parameters
ID_GOAL = 0
CAPACITY = 4
MAX_DELAY = 5000.0

# dataset specific parameters
UNIT_DIVISOR = 1000.0
MAX_RANGE = None
mgr = CarpoolDataManager('map_data/distance_matrix.csv', 'map_data/carpool_map_coordinates.csv')
mgr.filter_data(MAX_RANGE, ID_GOAL)


def constrain_solution(solution, data_manager):
    # if the solution produced by the pointer network surpasses a certain distance threshold,
    # constrain it so that it is within the threshold

    n = len(solution)

    solution_new = [solution[IDX_START].copy()]

    goal = solution[IDX_GOAL].copy()
    route_len = 0.0
    d_thr = solution[IDX_THR][0] * UNIT_DIVISOR
    d_start_goal = data_manager.distances_pts([solution[IDX_START], solution[IDX_GOAL]])
    delay = d_thr - d_start_goal[0][1]

    for i in range(IDX_START, n - 1):
        d = data_manager.distances_pts([solution[i], solution[i + 1], goal])

        if route_len + d[0][1] + d[1][2] <= d_thr:
            route_len += d[0][1]
            solution_new.append(solution[i+1].copy())
        else:
            route_len += d[0][2]
            break

    solution_new.append(goal)

    return solution_new, route_len, delay, d_start_goal[0][1]


def reward(solution, USE_CUDA=False):
    batch_size = solution[0].size(0)

    rewards = Variable(torch.zeros([batch_size]))

    ids_sol = [mgr.tensor_pts2ids(x.cpu()) for x in solution[IDX_START:]]
    n = len(ids_sol)

    ids_goal = mgr.tensor_pts2ids(solution[IDX_GOAL].cpu())

    d_thr = solution[IDX_THR][:, 0].cpu() * UNIT_DIVISOR
    route_len = torch.zeros(batch_size)
    incl = torch.ones(batch_size).bool()

    for i in range(n - 1):
        d_sa = mgr.tensor_distances_ids(ids_sol[i], ids_sol[i + 1])
        d_sg = mgr.tensor_distances_ids(ids_sol[i], ids_goal)
        d_sg_next = mgr.tensor_distances_ids(ids_sol[i + 1], ids_goal)

        dist = route_len + d_sa + d_sg_next

        excl = incl & (dist > d_thr)
        incl[excl] = False

        route_len[incl] += d_sa[incl]
        rewards[incl] += 1.0 - d_sa[incl] / d_thr[incl]

        rewards[excl] += 1.0 - d_sg[excl] / d_thr[excl]

    rewards[incl] += 1.0 - d_sg[incl] / d_thr[incl]

    if USE_CUDA:
        rewards = rewards.cuda()

    return - rewards


def generate_sample(num_points, data_manager, seed=None):
    data, _, dist = data_manager.sample_data(num_points, idx_goal=ID_GOAL, seed=seed)
    data = mgr.normalize(data)
    data = data.flatten()

    delay = np.random.rand() * MAX_DELAY
    d_thr = (dist[1, 0] + delay) / UNIT_DIVISOR
    data = np.insert(data, [IDX_THR], [d_thr, -1])

    return data


def create_dataset(size,
                   num_points,
                   data_dir,
                   seed=None,
                   purpose='train',
                   data_manager=mgr):

    if seed is not None:
        np.random.seed(seed)

    fname = 'carpool-size-{}-len-{}-{}.txt'.format(size, num_points, purpose)

    path = os.path.join(data_dir, fname)

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    else:
        if os.path.exists(path):
            return path

    dataset_file = open(path, 'w')

    def to_string(x):
        line = ''
        for j in range(len(x) - 1):
            line += '{} '.format(x[j])
        line += str(x[-1]) + '\n'
        return line

    print('Creating data set for {}...'.format(fname))

    for i in trange(size):
        x = generate_sample(num_points, data_manager)
        dataset_file.write(to_string(x))

    dataset_file.close()

    return fname


class CarpoolDataset(Dataset):

    def __init__(self, dataset_fname=None, train=False, size=20, num_samples=10000, seed=None, data_manager=mgr):
        super(CarpoolDataset, self).__init__()

        if seed is not None:
            np.random.seed(seed)

        print('Loading data into memory')
        self.data_set = []
        if not train:
            print('Loading data from file')
            with open(dataset_fname, 'r') as dset:
                for l in tqdm(dset):
                    x = np.array(l.split(), dtype=np.float32).reshape([-1, 2]).T
                    self.data_set.append(x)
        else:
            print('Generating new data')
            for l in tqdm(range(num_samples)):
                x = generate_sample(size, data_manager)
                x = x.reshape([-1, 2]).T
                self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]

# TEST DATA SEEDS
# --------------------------
# len 20: seed = 123456789
# len 30: seed = 1234567890
# len 50: seed = 246802468
# len 100: seed = 2468024680


if __name__ == '__main__':
    mgr_test = CarpoolDataManager('map_data/distance_matrix_test.csv',
                                  'map_data/carpool_map_coordinates_test.csv')

    p = 'testql'
    create_dataset(10000, 20, 'test_data/carpool', seed=123456789, data_manager=mgr_test, purpose=p)
    create_dataset(10, 20, 'test_data/carpool', data_manager=mgr_test, purpose=p)

    create_dataset(10000, 30, 'test_data/carpool', seed=1234567890, data_manager=mgr_test, purpose=p)
    create_dataset(10, 30, 'test_data/carpool', data_manager=mgr_test, purpose=p)

    create_dataset(10000, 50, 'test_data/carpool', seed=246802468, data_manager=mgr_test, purpose=p)
    create_dataset(10, 50, 'test_data/carpool', data_manager=mgr_test, purpose=p)

    create_dataset(10000, 100, 'test_data/carpool', seed=2468024680, data_manager=mgr_test, purpose=p)
    create_dataset(10, 100, 'test_data/carpool', data_manager=mgr_test, purpose=p)
