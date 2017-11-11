import unittest
import numpy as np
import hough_transform
import processor


def generate_line_points(m, im_len):
    """ Generate test line points on an image where origin is top left """
    run = int(im_len / abs(m))
    rise = run * abs(m)

    x_points = np.arange(0, run + 0).astype(np.uint8)
    start = rise - 1
    end = 0
    if m < 0:
        end = start
        start = 0
    y_points = np.linspace(start, end, len(x_points)).astype(np.uint8)

    return x_points, y_points


def plot_image(img):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(img)
    plt.axis('equal')
    plt.show()


class TestHoughLine(unittest.TestCase):
    def test_negative_gradient(self):
        im_len = 100
        gradient = -3.3
        x_points, y_points = generate_line_points(m=gradient, im_len=im_len)

        img = np.zeros((im_len, im_len), dtype=np.uint8)
        img[y_points, x_points] = 255

        accumulator, thetas, rhos = hough_transform.hough_line(img)
        idx, theta, rho = processor.peak_votes(accumulator, thetas, rhos)

        self.assertEqual(idx, 25453)
        self.assertTrue(np.isclose(
            processor.theta2gradient(theta), gradient, 1e-2))
        self.assertTrue(np.isclose(
            processor.rho2intercept(theta, rho), -1.71624))

        # print("idx={}, rho={:.2f}, theta={:.0f} gradient={:.2f} intercept={:.5f}".format(
        #     idx, rho, np.rad2deg(theta), processor.theta2gradient(theta), processor.rho2intercept(theta, rho)))

    def test_positive_gradient(self):
        im_len = 80
        gradient = 5.1
        x_points, y_points = generate_line_points(m=gradient, im_len=im_len)

        img = np.zeros((im_len, im_len), dtype=np.uint8)
        img[y_points, x_points] = 255

        accumulator, thetas, rhos = hough_transform.hough_line(img)
        idx, theta, rho = processor.peak_votes(accumulator, thetas, rhos)

        self.assertEqual(idx, 22961)
        self.assertTrue(np.isclose(
            processor.theta2gradient(theta), gradient, 1e-2))
        self.assertTrue(np.isclose(
            processor.rho2intercept(theta, rho), y_points[0], 0.1))

    def test_vertical_line(self):
        img = np.zeros((31, 31), dtype=np.uint8)
        img[1, 15] = 255
        img[5, 15] = 255
        img[10, 15] = 255
        img[15, 15] = 255
        img[20, 15] = 255
        img[25, 15] = 255
        img[30, 15] = 255

        accumulator, thetas, rhos = hough_transform.hough_line(img)
        idx, theta, rho = processor.peak_votes(accumulator, thetas, rhos)

        self.assertEqual(idx, 10710)
        self.assertEqual(round(rho), 16)
        self.assertEqual(theta, 0)


if __name__ == '__main__':
    unittest.main()
