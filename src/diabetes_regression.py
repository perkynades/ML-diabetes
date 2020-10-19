import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression

x, y = load_diabetes(return_X_y=True)
