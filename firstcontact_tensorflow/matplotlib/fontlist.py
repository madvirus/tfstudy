import matplotlib
matplotlib.use('TkAgg')

import matplotlib.font_manager
fontNames = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
for n in fontNames:
    print(n)
