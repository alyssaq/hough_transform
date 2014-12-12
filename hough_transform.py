import numpy as np
from scipy import misc

def hough_line(img):
  """
  Hough transform for lines

  Input:
  img - 2D binary image with nonzeros representing edges

  Returns:
  accumulator - 2D array of the hough transform accumulator
  theta - array of angles used in computation, in radians.
  rhos - array of rho values. Max size is 2 times the diagonal
         distance of the input image.
  """
  # theta = -90 to 89, rho = -diag_len to diag_len
  thetas = np.linspace(-np.pi/2, np.pi/2, 90)
  num_thetas = len(thetas)
  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)

  width, height = img.shape
  diag_len = np.ceil(np.sqrt(width * width + height * height))
  rhos = np.arange(-diag_len, diag_len)

  # accumulator array of theta vs rho
  accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
  x_idxs, y_idxs = np.nonzero(img)

  for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]

    for t_idx in range(num_thetas):
      rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
      print x, y, rho, thetas[t_idx] * 180/np.pi
      accumulator[rho, t_idx] += 1

  return accumulator, thetas, rhos

def show_hough_line(img, accumulator):
  import matplotlib.pyplot as plt

  plt.subplot(1, 2, 1)
  plt.title('Input image')
  plt.imshow(img, 'gray')
  plt.subplot(1, 2, 2)
  plt.title('Hough transform')
  plt.imshow(accumulator, cmap='jet')
  plt.axis('off')
  plt.savefig('imgs/output.png', bbox_inches='tight')
  plt.show()

if __name__ == '__main__':
  imgpath = 'imgs/binary_crosses.png'
  img = misc.imread(imgpath)
  accumulator, thetas, rhos = hough_line(img)
  show_hough_line(img, accumulator)
