import json, re, sys
stats = json.load(open(sys.argv[1]))
for path in sys.argv[2:]:
    t = open(path).read()
    def rep(m):
        d, k = m.group(1), m.group(2)
        return str(stats[d][k])
    t2, n = re.subn(r"@@(MIND|EBNeRD)\.([a-z_]+)@@", rep, t)
    open(path, "w").write(t2); print(path, n)
