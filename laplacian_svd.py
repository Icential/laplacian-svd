import argparse
import cv2 as cv
from skimage.graph import pixel_graph
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import SparsePCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.sparse.csgraph as csg

def size_reducer(img, biggest_length):
    h, w = img.shape[:2]
    if max(h, w) > biggest_length:
        if h > w:
            new_h = biggest_length
            new_w = int(w * biggest_length / h)
        else:
            new_w = biggest_length
            new_h = int(h * biggest_length / w)
        img = cv.resize(img, (new_w, new_h))
    return img

def main(args):
    img = cv.imread(args.img, cv.IMREAD_GRAYSCALE)
    img = size_reducer(img, args.biggest_length)
    
    img_graph = pixel_graph(img, connectivity=2)
    img_graph = img_graph[0].astype(np.float32) 
    laplacian = csg.laplacian(img_graph, normed=True)

    pca = TruncatedSVD(n_components=3)

    # image = m*n. laplacian = (m*n) * (m*n) * 32 bytes
    reconst_laplacian = pca.inverse_transform(pca.fit_transform(laplacian)) # forced dense are you serious


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Input file path")
    parser.add_argument("--biggest_length", type=int, default=50, help="Resize the biggest length to this value")

    args = parser.parse_args()
    main(args)