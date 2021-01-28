import numpy as np

class Node:

  def __init__(self, id, is_leaf, **kwargs):
    self.parent = kwargs.get("parent", None)
    self.id = id
    self.is_leaf = is_leaf
    self.pivot = kwargs.get("pivot", None)
    self.median = kwargs.get("median", -1)
    if self.is_leaf:
      self.objects = kwargs.get("objects", [])
      self.file_path_s_1, self.file_path_s_2 = "", ""
    else:
      self.right = kwargs.get("right", None)
      self.left = kwargs.get("left", None)

  def set_parameters(self, pivot, median):
    self.pivot = pivot
    self.median = median

  def add_children(self, left, right):
    self.left = left
    self.right = right

  def add_objects(self, s_1, s_2):
    self.objects_left = s_1
    self.objects_right = s_2

  def save_leaf_objects_on_disk(self, file_path, s_1, s_2):
    self.file_path_s_1 = file_path + "_subset_1.npy"
    self.file_path_s_2 = file_path + "_subset_2.npy"
    np.save(self.file_path_s_1, np.array(s_1, dtype=object))
    np.save(self.file_path_s_2, np.array(s_2, dtype=object))

  def load_objects_from_disk(self, left=True, right=True):
    if left and not right:
      result = np.load(self.file_path_s_1, allow_pickle=True)
      return result
    if right and not left:
      result = np.load(self.file_path_s_1, allow_pickle=True)
      return result
    s_1 = np.load(self.file_path_s_1, allow_pickle=True)
    s_2 = np.load(self.file_path_s_2, allow_pickle=True)
    result = np.concatenate((s_1, s_2))
    return result

  def get_node_name(self):
    return self.id