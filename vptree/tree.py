import math
import time
import random
import numpy as np
from vptree.node import Node
import os
import json
from scipy.spatial import distance as d

class VPTree:

  def __init__(self, index_name, height, disk_mode=True, leaves_path=None, use_similarity=False):
    """by default the tree is built with euclidean distance and the leaves are saved on disk, 
    use_similarity=True allows you to use the cosine similarity
    use disk_mode=False if you want to keep all the tree in memory(not suggested for huge datasets)"""
    self.root = None
    self.index_name = index_name
    self.height = height #to review
    self.disk_mode = disk_mode
    self.leaves_path = leaves_path
    self.use_similarity=use_similarity
    self.distance_computed = 0
    self.file_accessed = 0
    self.file_created = 0

  def create_vptree(self, names_path, features_path):
    start = time.time()
    data = VPTree.read_data(names_path, features_path)
    n = len(data)
    print("Number of data:", n)
    max_height = math.floor(math.log(n,2)-1)
    print("The max height of the tree is:", max_height)
    if self.height > max_height: self.height = max_height
    self.distance_computed = 0
    #take 1 pivot randomly and set pivot as root
    self.root, s_1, s_2 = self.partition_by_median(data)
    print("Tree is building")
    self.create_tree_level(self.root, s_1, s_2, 1)
    end = time.time()
    print("Building of the tree completed in:", end-start, "s")
  
  def create_tree_level(self, node, s_1, s_2, iteration):
      is_leaf = iteration + 1 >= self.height
      left_node, s_1_left, s_2_left = self.partition_by_median(s_1, parent=node,is_left=True, is_leaf=is_leaf)
      right_node, s_1_right, s_2_right = self.partition_by_median(s_2, parent=node,is_left=False, is_leaf=is_leaf)
      node.add_children(right_node, left_node)
      if iteration + 1 < self.height:
        self.create_tree_level(left_node, s_1_left, s_2_left, iteration + 1)
        self.create_tree_level(right_node, s_1_right, s_2_right, iteration + 1)
      else:
        if self.disk_mode:
          left_path = self.get_leaves_path(left_node.get_node_name())
          right_path = self.get_leaves_path(right_node.get_node_name())
          left_node.save_leaf_objects_on_disk(left_path, s_1_left, s_2_left)
          right_node.save_leaf_objects_on_disk(right_path, s_1_right, s_2_right)
        else:
          left_node.add_objects(s_1_left, s_2_left)
          right_node.add_objects(s_1_right, s_2_right)

  def partition_by_median(self, data, parent=None,is_left=False,is_leaf=False):
    pivot_index = random.choice(range(len(data)))
    pivot = data[pivot_index]
    del data[pivot_index]
    #compute all the distances
    distances = np.array([self.compute_distance(pivot[1],element[1]) for element in data])
    #sort the distances
    zipped_data_distances = sorted(zip(data, distances), key= lambda x:x[1])
    ordered_data, distances = zip(*zipped_data_distances)
    median = np.median(distances)
    #get the median
    s_1 = [element for element, distance in zipped_data_distances if distance <= median]
    s_2 = [element for element, distance in zipped_data_distances if distance >= median]
    #update node
    if parent == None:
      node = Node(id="0", is_leaf=is_leaf, pivot=pivot, median=median)
    else:
      node_id = parent.id + str(0 if is_left else 1)
      node = Node(node_id, is_leaf=is_leaf, pivot=pivot, median=median)
    return node, s_1, s_2

  def knn_search(self, k, query):
    start = time.time()
    nn = [None for i in range(k)]
    d_nn = [math.inf for i in range(k)]
    self.distance_computed = 0
    self.file_accessed = 0
    nn, d_nn = self.search_subtree(self.root, nn, d_nn, k, query)
    end = time.time()
    print("Query answered in", end-start, " s")
    return self.reorder_list_on_distances(nn, d_nn, desc=False)

  def search_subtree(self, node, nn, d_nn, k, query):
    pivot, median = node.pivot, node.median
    distance = self.compute_distance(pivot[1], query)
    if distance < d_nn[0]:
      d_nn[0] = distance
      nn[0] = pivot
      nn, d_nn = self.reorder_list_on_distances(nn, d_nn)
    if node.is_leaf:
      return self.search_in_leaf(node, nn, d_nn, k, query)
    if distance - d_nn[0] <= median:
      nn, d_nn = self.search_subtree(node.left, nn, d_nn, k, query)
    if distance + d_nn[0] >= median:
      nn, d_nn = self.search_subtree(node.right, nn, d_nn, k, query)
    return nn, d_nn

  def search_in_leaf(self, node, nn, d_nn, k, query):
    objects = []
    distance_pivot = self.compute_distance(node.pivot[1], query)
    left, right = False, False
    if self.disk_mode:
      if distance_pivot - d_nn[0] <= node.median: 
        left = True
        self.file_accessed = self.file_accessed + 1
      if distance_pivot + d_nn[0] >= node.median: 
        right = True
        self.file_accessed = self.file_accessed + 1
      objects = node.load_objects_from_disk(left=left, right=right)
    else:
      objects = node.objects_left + node.objects_right
    for obj in objects:
      distance = self.compute_distance(obj[1], query)
      if distance < d_nn[0]:
        nn[0] = obj
        d_nn[0] = distance
        nn, d_nn = self.reorder_list_on_distances(nn, d_nn)
    return nn, d_nn

  def reorder_list_on_distances(self, nn, d_nn, desc=True):
      zipped = sorted(zip(nn, d_nn), key= lambda x:x[1], reverse=desc)
      nn, d_nn = zip(*zipped)
      return list(nn), list(d_nn)

  def get_leaves_path(self, file_name):
    if not self.leaves_path is None:
      directory = os.path.join(self.leaves_path, self.index_name)
    else: directory = os.path.join(os.curdir, self.index_name)
    if not os.path.exists(directory):
      os.mkdir(directory)
      print("directory created", directory)
    leaves_directory = os.path.join(directory, "leaves_"+ str(self.height))
    if not os.path.exists(leaves_directory):
      os.mkdir(leaves_directory)
    return os.path.join(leaves_directory, file_name)

  def compute_distance(self, a, b):
    self.distance_computed = self.distance_computed + 1
    if self.use_similarity:
      return 1 - np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return d.euclidean(a,b)

  @staticmethod
  def read_data(file_path_names, file_path_features):
    names = np.load(file_path_names)
    features = np.load(file_path_features)
    return [(name, feature) for name, feature in zip(names, features)]

  @staticmethod
  def print_tree(node, level, disk_mode=True):
    indentation = "\n" + str(level * "\t")
    response = "id: " + node.id + " " + str(node.pivot)
    if node.is_leaf:
      if disk_mode: 
        response += indentation + str(node.file_path_s_1)
        response += indentation + str(node.file_path_s_2)
      else:
        response += indentation + str(node.objects_left)
        response += indentation + str(node.objects_right)
      return response
    response += indentation + VPTree.print_tree(node=node.right, level=level+1, disk_mode=disk_mode)
    response += indentation + VPTree.print_tree(node=node.left, level=level+1, disk_mode=disk_mode)
    return response
  
  @staticmethod
  def load_vptree(path):
    if not os.path.exists:
      print("the path do not exist")
      return None
    entry_list=[]
    with open(path,'r', encoding='utf-8') as f:
      json_tree = json.load(f)
      entry_list=json_tree["nodes"]
    root_node=VPTree.parse_node('0',entry_list)
    index_name = json_tree["index"]
    height = json_tree["height"]
    use_similarity = json_tree.get("use_similarity", False)
    vp_tree = VPTree(index_name=index_name,height=height,leaves_path=path, 
                      use_similarity=use_similarity)
    vp_tree.root = root_node
    print("Tree loaded correctly")
    return vp_tree
  
  @staticmethod
  def parse_node(id, nodes):
    node_json = None
    for element in nodes:
      if element["id"]==id:
        node_json = element
    node=Node(id=node_json["id"], is_leaf=node_json["is_leaf"], 
              pivot=node_json["pivot"], median=node_json["median"])
    if (node.is_leaf):
      node.file_path_s_1=node_json["left_file"]
      node.file_path_s_2=node_json["right_file"]
    else:
      right=VPTree.parse_node(node_json["right_child"],nodes)
      left=VPTree.parse_node(node_json["left_child"],nodes)
      node.add_children(left, right)
    return node

  @staticmethod
  def save_vptree(file_path, tree):
    if not os.path.exists(file_path): os.mkdir(file_path)
    file = os.path.join(file_path, tree.index_name + '.json')
    if os.path.exists(file):
      os.remove(file)
    with open(file, 'a') as json_file:
      index_json = {"index": tree.index_name, "nodes":[], 
                    "height":tree.height, "use_similarity": tree.use_similarity}
      VPTree.save_node(tree.root, index_json)
      vp_tree_json = json.dumps(index_json, cls=NumpyEncoder)
      json_file.write(vp_tree_json)
      print("File saved correctly in:", file)
    return file
  
  @staticmethod
  def save_node(node, index_json):
    if node.is_leaf:
        row_json={"is_leaf":True, 
                    "id":node.id,
                    "pivot" : node.pivot,
                    "median":node.median, 
                    "left_file":node.file_path_s_1, 
                    "right_file":node.file_path_s_2}
        index_json["nodes"].append(row_json)
    else:
        row_json={"is_leaf":False,
                  "id":node.id, 
                  "pivot": node.pivot,
                  "median": node.median,
                  "right_child":node.right.id,
                  "left_child":node.left.id}
        index_json["nodes"].append(row_json)
        VPTree.save_node(node.left, index_json)
        VPTree.save_node(node.right,index_json)


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if obj is None:
            return ""
        return json.JSONEncoder.default(self, obj)