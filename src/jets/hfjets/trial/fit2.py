import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#bkg_par = [a,b,c]
def func(x, *par):
    #return a * np.exp(-b * x) + c
    return par[0] * np.exp(-par[1] * x) + par[2]

def gaussian(x, *par):
    return par[0] * np.exp(-(x-par[1])**2 / par[2])

def sig_bkg(x, a,b,c,d,e,f):
    return gaussian(x,a,b,c) + func(x,d,e,f)

xdata = np.linspace(0, 4, 50) 
bkg_par = [2.5, 1.3, 0.5]
y = func(xdata, *bkg_par)
np.random.seed(1729)
y_noise = 0.05 * np.random.normal(size=xdata.size)
y_signal = gaussian(xdata, 2.5, 1.5, 0.5)
ydata = y + y_noise + y_signal
plt.plot(xdata, ydata, 'b+', label='data')



popt, pcov = curve_fit(sig_bkg, xdata, ydata)

plt.plot(xdata, sig_bkg(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt[0:3]))


#popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
plt.plot(xdata, gaussian(xdata, *popt[0:3]), 'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt[0:3]))
plt.plot(xdata, func(xdata, *popt[3:6]), 'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt[3:6]))

plt.plot(xdata, ydata, 'b+', label='data')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
