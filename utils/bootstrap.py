import numpy as np
def bootstrap(length):
    r = np.arange(0,length)
    np.random.shuffle(r)
    sr = r
    u = sr[:int(length * 0.63)]
    l = sr[int(length * 0.63):]
    print(len(l))
    print(len(u))
    d = np.random.choice(u,int(length*0.37))
    b = np.concatenate((d,u))
    return b,l