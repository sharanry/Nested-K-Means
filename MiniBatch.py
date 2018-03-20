import numpy as np
import matplotlib.pyplot as plt


class MiniBatchKMeans():
    def __init__(self, x, k, b):
        self.x = x
        self.k = k
        self.b = b
        self.N, self.ndim = x.shape
        self.initialise_cSv()
        self.convCritirion = False

    def initialise_cSv(self):
        # Algorithm 2

        choice = np.random.choice(self.N, self.k)

        self.c = self.x[choice]
        self.S = self.x[choice]
        self.v = np.ones([self.k, 1], dtype=np.int32)
        return self.c, self.S, self.v

    def a(self, i):
        # argmin of euclidean distance

        return (np.linalg.norm(self.c.reshape(-1, self.ndim) - self.x[i], axis=1)).argmin()



    def accumulate(self, i):
        # Algorithm 3
        am = self.a(i)
        self.S[am] += self.x[i]
        self.v[am] += 1

    def mbatch(self, max_iter):
        # Algorithm 4

        iter = 0
        while not self.convCritirion and iter < max_iter:
            temp = self.c
            M = np.random.choice(self.N, self.b)
            for i in M:
                self.accumulate(self.a(i))

            self.c = self.S / self.v.reshape(-1, 1)

            if (temp - self.c == 0).all():
                self.convCritirion = True
            iter += 1


    def show(self, keep = False):
        C = [self.a(i) for i in range(self.N)]
        plt.cla()
        plt.scatter(self.x[:,0], self.x[:,1], c=C)
        plt.plot(self.c[:,0],self.c[:,1],'x', color="black",markersize=10)
        plt.draw()
        if keep :
            plt.ioff()
        plt.show()
