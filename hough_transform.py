import numpy as np
from scipy import misc

def hough_line(img):
  # theta = -90 to 89, rho = -diag_len to diag_len
  thetas = np.linspace(-np.pi/2, np.pi/2, 180)
  num_thetas = len(thetas)
  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)

  width, height = img.shape
  diag_len = np.ceil(np.sqrt(width * width + height * height))

  # accumulator array of theta vs img
  accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
  x_idxs, y_idxs = np.nonzero(img)

  for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]

    for theta in range(num_thetas):
      rho = round(x * cos_t[theta] + y * sin_t[theta]) + diag_len
      accumulator[rho, theta] += 1

  return accumulator, thetas

def show_hough_line(imgpath):
  import matplotlib.pyplot as plt

  img = misc.imread(imgpath)
  accumulator, thetas = hough_line(img)
  plt.imshow(accumulator, cmap='jet')
  plt.show()

if __name__ == '__main__':
  show_hough_line('imgs/binary_crosses.png')
