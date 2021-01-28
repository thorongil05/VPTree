from vptree.tree import VPTree
from scipy.spatial import distance as d
import numpy as np

def test_partition_by_median_1():
    data = [("image0", np.array([0,2,1])) , ("image1",np.array([2,3,6])), ("image2",np.array([5,3,2])),
          ("image3",np.array([5,6,4])), ("image4",np.array([5,16,1])), ("image5",np.array([2,6,2])), ("image6",np.array([1,3,1]))]
    vantage_point_tree = VPTree("Index_Test", 5)
    node, s_1, s_2 = vantage_point_tree.partition_by_median(data)
    return data, node, s_1, s_2


def test_partition_by_median_2():
    data = [("img_1",np.array([0,0])), ("img_2",np.array([0,1])), ("img_3",np.array([2,2])), ("img_4",np.array([3,3]))]
    vantage_point_tree = VPTree("Index_Test", 5)
    node, s_1, s_2 = vantage_point_tree.partition_by_median(data)
    return data, node, s_1, s_2

def validate_results(data, node, s_1, s_2):
    pivot = node.pivot
    data_without_pivot = np.array([element for element in data if element[0] != pivot[0]], dtype=object)
    distances = [d.euclidean(pivot[1], element[1]) for element in data_without_pivot]
    distances = sorted(distances)
    if np.median(distances) == node.median: 
        return True, "Correct Median" 
    else: return False, "Median is not correct"

def test():
    print("\tPartition By Median Tests\n")
    data, node, s_1, s_2 = test_partition_by_median_1()
    test_result, message = validate_results(data, node, s_1, s_2)
    print("\t\tTest 1: \t", "passed" if test_result else "Failed: " + message)
    data, node, s_1, s_2 = test_partition_by_median_2()
    test_result, message = validate_results(data, node, s_1, s_2)
    print("\t\tTest 2: \t", "passed" if test_result else "Failed: " + message)