# SIFT-GSC
SIFT-Based Genetic Seam Carving (SIFT-GSC) is an algorithm inspired by Surf-Based Genetic Seam Carving [[1]](#1).

It resizes images based on Seam Carving and SIFT feature descriptors, using genetic algorithm to explore the solution  
space.

## Install
To install the dependencies (with python 3.8.8 tested):
```
$ pip install opencv-python==4.5.1.48 numpy==1.20.2 scikit-learn==0.24.1 scikit-image==0.18.1
```

## Usage
To run a single image (changing the main function parameters):
```
$ python main.py
```

To run the images on dataset directory (changing the main function parameters and the desired windows sizes at line 9):
```
$ python get_results.py
```

## References
<a id="1">[1]</a>
Oliveira, S. A., Rocha Neto, A. R., and Bezerra, F. N. (2016).
A novel genetic  algorithms and surf-based approach for image retargeting.
Expert Systems  with Applications, 44:332–343.
