import unittest
import cProfile
import numpy as np
import sys
sys.path.append('../')
from VDBMethods.KDTree import *
from VDBMethods.SimpleDict import *

class VDBUnitTest(unittest.TestCase):

    def test_all(self):
        
        dim = 1

        def rand_point(dim):
            return [np.random.uniform(-1, 1) for d in range(dim)]

        '''points = [rand_point(dim) for x in range(10000)]
        additional_points = [rand_point(dim) for x in range(100)]
        query_points = [rand_point(dim) for x in range(100)]'''

        '''points = [rand_point(dim) for x in range(11)]
        additional_points = [rand_point(dim) for x in range(2)]
        query_points = [rand_point(dim) for x in range(2)]'''

        points = [[2],[3],[4],[5],[6]]
        additional_points = [[1],[7]]
        query_points = [[4.9]]

        def dist_sq_func(a, b):
            return sum((x - b[i]) ** 2 for i, x in enumerate(a))

        def get_knn_naive(points, point, k, return_dist_sq=True):
            neighbors = []
            for i, pp in enumerate(points):
                dist_sq = dist_sq_func(point, pp)
                neighbors.append((dist_sq, pp))
            neighbors = sorted(neighbors)[:k]
            return neighbors if return_dist_sq else [n[1] for n in neighbors]

        def get_nearest_naive(points, point, return_dist_sq=True):
            nearest = min(points, key=lambda p:dist_sq_func(p, point))
            if return_dist_sq:
                return (dist_sq_func(nearest, point), nearest) 
            return nearest


        kd_tree_results = []
        naive_results = []
        
        global test_and_bench_kd_tree
        global test_and_bench_naive

        def test_and_bench_kd_tree():
            nn_count = 8
            global kd_tree
            kd_tree = KDTree(points, dim)
            for point in additional_points:
                kd_tree.add_point(point)
            #kd_tree_results.append(tuple(kd_tree.get_knn([0] * dim, nn_count)))
            for t in query_points:
                kd_tree_results.append(tuple(kd_tree.get_knn(t, 3)))
            for t in query_points:
                kd_tree_results.append(tuple(kd_tree.get_nearest(t)))

        def test_and_bench_naive():
            nn_count = 8
            all_points = points + additional_points
            #naive_results.append(tuple(get_knn_naive(all_points, [0] * dim, nn_count)))
            for t in query_points:
                naive_results.append(tuple(get_knn_naive(all_points, t, 3)))
            for t in query_points:
                naive_results.append(tuple(get_nearest_naive(all_points, t)))

        print("Running KDTree...")
        cProfile.run("test_and_bench_kd_tree()")
        
        print("Running naive version...")
        cProfile.run("test_and_bench_naive()")

        print("Query results same as naive version?: {}"
            .format(kd_tree_results == naive_results))
        
        print('********************************')
        print(kd_tree_results)
        print('********************************')
        print(naive_results)

        self.assertEqual(kd_tree_results,naive_results, msg = "Query results mismatch")
        
        self.assertEqual(len(list(kd_tree)), len(points) + len(additional_points),msg = "Query results mismatch")

if __name__ == '__main__':
    unittest.main()