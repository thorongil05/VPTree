from vptree.tree import VPTree

IDS_PATH_1 = "."
FEATURES_PATH_1 = "."
IDS_PATH_2 = "."
FEATURES_PATH_2 = "."
IDS_PATH_3 = "."
FEATURES_PATH_3 = "."
IDS_PATH_4 = "."
FEATURES_PATH_4 = "."

# Create Tree Test 1
vantage_point_tree = VPTree("Index_Test_1",4)
vantage_point_tree.create_vptree(IDS_PATH_1,FEATURES_PATH_1)

# Create Tree Test 2
vantage_point_tree = VPTree("Index_Test_2",4)
vantage_point_tree.create_vptree(IDS_PATH_2, FEATURES_PATH_2)

# Create Tree Test 3
vantage_point_tree = VPTree("Index_Test_3",6)
vantage_point_tree.create_vptree(IDS_PATH_3, FEATURES_PATH_3)

# Create Tree Test 4
vantage_point_tree = VPTree("Index_Test_4",4, disk_mode=True)
vantage_point_tree.create_vptree(IDS_PATH_4, FEATURES_PATH_4)