import math
import numpy



def psnr(img1, img2):
    '''
    note: img should be in uint8
    '''
    mse = np.mean((img1 -img2) ** 2)

    if mse == 0:
        return 1e+6

    peak = 255.0 ** 2

    return 10 * math.log10(peak / mse)