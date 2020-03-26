#Visualize

import matplotlib.pyplot as plt
import numpy as np

x=np.arange(1,5)
y=x**3
plt.figure(figsize=(10,3))
plt.plot([1,2,3,4],[1,4,9,16],'go', x, y, 'r^')
plt.title('Tot de Macht')
plt.xlabel('eenheden')
plt.ylabel('kwadraat')
plt.show()