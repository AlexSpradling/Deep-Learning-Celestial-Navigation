import shutil
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def build_data_source():
  # move files from cloud_cover2 to cloud_cover1
  files_1_dir = 'projects\capstone\cloud_cover1'
  files_2_dir = 'projects\capstone\cloud_cover2'
  files_1 = os.listdir(files_1_dir)
  files_2 = os.listdir(files_2_dir)

  files_1 = [f for f in files_1 if f.endswith('.png')]
  files_2 = [f for f in files_2 if f.endswith('.png')]

  for file_name in files_2:
      src_path = os.path.join(files_2_dir, file_name)
      dst_path = os.path.join(files_1_dir, file_name)
      if os.path.exists(src_path):
          shutil.move(src_path, dst_path)

  # now everything is consolidated, source list of file names
  files = os.listdir('projects\capstone\cloud_cover1')
  files = [f for f in files if f.endswith('.png')]
  files.sort()

  # split into training and validation lists
  train_list, test_list = train_test_split(files, train_size = 0.9, shuffle = False)

  train_list.sort()
  test_list.sort()
  print(len(train_list), len(test_list))

  #Create the train and valid directories
  train_dir = '../images/train'
  valid_dir = '../images/valid'
  os.makedirs(train_dir, exist_ok=True)
  os.makedirs(valid_dir, exist_ok=True)

  src_dir = 'projects\capstone\cloud_cover1'
  # Move the files for the train set
  for file_name in train_list:
      src_path = os.path.join(src_dir, file_name)
      dst_path = os.path.join(train_dir, file_name)
      if os.path.exists(src_path):
          shutil.move(src_path, dst_path)

  # Move the files for the valid set
  for file_name in test_list:
      src_path = os.path.join(src_dir, file_name)
      dst_path = os.path.join(valid_dir, file_name)
      if os.path.exists(src_path):
          shutil.move(src_path, dst_path)
  return files

data = build_data_source()