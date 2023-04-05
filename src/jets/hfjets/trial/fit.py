import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / wid)

def sig_bkg(x, a, b, c, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / wid) +  a* np.exp(-b * x) + c

xdata = np.linspace(0, 4, 50) 
y = func(xdata, 2.5, 1.3, 0.5)
np.random.seed(1729)
y_noise = 0.05 * np.random.normal(size=xdata.size)
y_signal = gaussian(xdata, 2.5, 1.5, 0.5)
ydata = y + y_noise + y_signal
plt.plot(xdata, ydata, 'b+', label='data')

popt, pcov = curve_fit(sig_bkg, xdata, ydata)
plt.plot(xdata, sig_bkg(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt[0:3]))


#popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
plt.plot(xdata, func(xdata, *popt[0:3]), 'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt[0:3]))
plt.plot(xdata, gaussian(xdata, *popt[3:6]), 'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt[3:6]))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print(popt)
