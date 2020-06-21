import argparse
import googlemaps
import carpool_data as cd
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get Distance Matrix from Coordinates")

    parser.add_argument('--api_key',  default='')
    parser.add_argument('--coords_file', default='../map_data/carpool_map_coordinates_test.csv')
    parser.add_argument('--mode', default='driving')
    parser.add_argument('--units', default='metric')
    parser.add_argument('--language', default='en')

    args = vars(parser.parse_args())

    # load location coordinates into array
    points = cd.load_coordinates(args['coords_file'])
    points = np.asarray(points)
    num_points = len(points)

    # in order to read data by hundreds (due to internet connection speeds)
    limit = num_points // 100 * 100
    remainder = num_points % 100

    gmaps = googlemaps.Client(key=args['api_key'])

    distances = []
    durations = []
    for i in range(num_points):
        distances.append([])
        durations.append([])
        j = 0
        incr = 100
        while j < num_points:

            if j >= limit:
                incr = remainder

            dist_mat_rows = gmaps.distance_matrix(points[i:i + 1, :], points[j:j + incr, :],
                                                  mode=args['mode'],
                                                  units=args['units'],
                                                  language=args['language'])['rows']

            for element in dist_mat_rows[0]['elements']:
                distances[i].append(element['distance']['value'])
                durations[i].append(element['duration']['value'])

            j += incr

    distance_matrix = np.asarray(distances, dtype=np.uint32)
    duration_matrix = np.asarray(durations, dtype=np.uint32)

    np.savetxt('map_data/distance_matrix_test.csv', distance_matrix, fmt='%d', delimiter=',', newline='\n')
    np.savetxt('map_data/duration_matrix_test.csv', duration_matrix, fmt='%d', delimiter=',', newline='\n')
