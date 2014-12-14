import numpy as np
from scipy import misc

def hough_line(img, angle_step=1):
  """
  Hough transform for lines

  Input:
  img - 2D binary image with nonzeros representing edges
  angle_step - Spacing between angles to use every n-th angle
    between -90 and 90 degrees. Default step is 1.

  Returns:
  accumulator - 2D array of the hough transform accumulator
  theta - array of angles used in computation, in radians.
  rhos - array of rho values. Max size is 2 times the diagonal
         distance of the input image.
  """
  # theta = -90 to 89, rho = -diag_len to diag_len
  thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
  num_thetas = len(thetas)
  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)

  width, height = img.shape
  diag_len = np.ceil(np.sqrt(width * width + height * height))
  rhos = np.arange(-diag_len, diag_len)

  # accumulator array of theta vs rho
  accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
  y_idxs, x_idxs = np.nonzero(img)

  for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]

    for t_idx in range(num_thetas):
      rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
      accumulator[rho, t_idx] += 1

  return accumulator, thetas, rhos

def show_hough_line(img, accumulator):
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(1, 2, figsize=(10, 10))

  ax[0].imshow(img, cmap=plt.cm.gray)
  ax[0].set_title('Input image')
  ax[0].axis('image')

  ax[1].imshow(
    accumulator, cmap='jet',
    extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
  ax[1].set_aspect('equal', adjustable='box')
  ax[1].set_title('Hough transform')
  ax[1].set_xlabel('Angles (degrees)')
  ax[1].set_ylabel('Distance (pixels)')
  ax[1].axis('image')

  #plt.axis('off')
  plt.savefig('imgs/output.png', bbox_inches='tight')
  plt.show()

if __name__ == '__main__':
  imgpath = 'imgs/binary_crosses.png'
  img = misc.imread(imgpath)
  accumulator, thetas, rhos = hough_line(img)
  show_hough_line(img, accumulator)
