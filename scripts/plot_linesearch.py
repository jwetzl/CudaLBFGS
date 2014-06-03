import sys
import os
import subprocess

if len(sys.argv) < 2:
    sys.stderr.write('Usage: %s iteration-number\n' % sys.argv[0])
    sys.exit(1)

fn = 'linesearch_iteration_%s.txt' % sys.argv[1]

if not os.path.exists(fn):
    sys.stderr.write('File \'%s\' does not exist.\n' % fn)
    sys.exit(1)

plotscript = """# Temporary plot to get min and max
plot '%s' using 1:2
MIN=GPVAL_X_MIN
MAX=GPVAL_X_MAX

plot '%s' using 1:2 pt 7 notitle, \
'%s' using 1:2:($0+1) with labels offset 2 notitle, \
'%s' using 1:2:((MAX-MIN)*0.05):((MAX-MIN)*0.05*$3) with vectors head filled lt 3 notitle
""" % (fn, fn, fn, fn)

with open('plot.gp', 'w') as f:
    f.write(plotscript)

subprocess.call(['gnuplot', '-p', 'plot.gp'])