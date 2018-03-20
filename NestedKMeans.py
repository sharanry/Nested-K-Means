import numpy as np
import matplotlib.pyplot as plt


class NestedKMeans():
    def __init__(self, x, k, b, rho=100, earlyStop=False):
        self.x = x
        self.k = k
        self.b = b
        self.N, self.ndim = x.shape
        self.convCritirion = False
        self.d = np.zeros(self.N)
        self.a = np.zeros(self.N, dtype=np.int32) - 1
        # self.a = np.random.choice(self.k, self.N)
        self.l = np.zeros([self.N, self.k]) - 1
        self.rho = rho
        self.earlyStop = earlyStop

    def assign(self, i):
        # argmin of euclidean distance
        # print(np.linalg.norm(self.c.reshape(-1, self.ndim) - self.x[i], axis=0))
        return (np.linalg.norm(self.c.reshape(-1, self.ndim) - self.x[i])).argmin()

    def assignment_with_bounds(self, i):
        self.d[i] = np.linalg.norm(self.c[self.assign(i)] - self.x[i]).min()



        for j in range(self.k):
            if j != self.a[i]:
                if self.l[i, j] < self.d[i]:
                    self.l[i, j] = np.linalg.norm(self.c[j] - self.x[i])
                    if self.l[i, j] < self.d[i]:
                        self.a[i] = j
                        self.d[i] = self.l[i, j]

    def initialise_cSv(self):
        # Algorithm 2

        self.c = np.zeros([self.k, self.ndim])
        self.S = np.zeros([self.k, self.ndim])
        self.v = np.zeros(self.k, dtype=np.int32)
        for j in range(self.k):
            choice = np.random.choice(self.N)
            self.c[j] = self.x[choice]
            self.S[j] = self.x[choice]
            self.v[j] = 1
        # return self.c, self.S, self.v
        print(self.v)
    def accumulate(self, i):
        # Algorithm 3
        am = self.a[i]
        self.S[am] += self.x[i]
        self.v[am] += 1

    def train(self, max_iter):
        # Algorithm 5
        t = 1
        M0 = 0
        M1 = self.b
        self.initialise_cSv()
        # print("0 c ", self.c)
        # print("0 S ", self.S)
        # print("0 v ", self.v)
        # print("0 x ", self.x)
        sse = np.zeros(self.k)
        p = np.zeros(self.k) # where to initialize??
        while True:
            for i in range(M0):
                for j in range(self.k):
                    self.l[i, j] -= p[j]

            a_old = np.zeros(self.N, dtype=np.int32) - 1
            for i in range(M0):
                a_old[i] = self.a[i]
                sse[a_old[i]] -= self.d[i]**2
                self.S[a_old[i]] -= self.x[i]
                self.v[a_old[i]] -= 1
                self.assignment_with_bounds(i)
                self.accumulate(i)
                sse[self.a[i]] += self.d[i]**2

            for i in range(M0, M1):
                for j in range(self.k):
                    self.l[i, j] = np.linalg.norm(self.x[i] - self.c[j])
            # print("1 l ", self.l)
            for i in range(M0, M1):
                self.a[i] = self.l[i,:].argmin()
                # print("id: ", i, " assign: ", self.a[i])
                self.d[i] = self.l[i, self.a[i]]
                self.accumulate(i)
                sse[self.a[i]] += self.d[i]**2
            # print("2 a ", self.a)
            # print("2 d ", self.d)
            # print("2 S ", self.S)
            # print("2 v ", self.v)

            c_old = np.zeros([self.k, self.ndim]) - 1
            sigmaC = np.zeros(self.k) - 1
            for j in range(self.k):
                sigmaC[j] = np.sqrt(sse[j]/self.v[j]*(self.v[j]-1))
                c_old[j] = self.c[j]
                self.c[j] = self.S[j]/self.v[j]
                p[j] = np.linalg.norm(self.c[j] - c_old[j])

            # print(sigmaC/p)
            # print("p: ", p)
            # print("sigmaC: ", sigmaC)
            # print("self.v: ", self.v)
            # print("sse: ", sse)

            if (p == 0).all() and M1 == self.N:
                print("Convergence Criterion Satisfied")
                break

            if self.earlyStop and M1==self.N:
                print("earlyStop")
                break
            if self.convCritirion or t > max_iter :
                if M1 == self.N:
                    break

            if np.nanmin(sigmaC/p) > self.rho:
                M0 = M1
                M1 = min(2 * M1, self.N)
                print("t: ", t, "M1: ", M1)


            t += 1



    def show(self, keep=False):
        # C = self.a
        plt.cla()
        plt.scatter(self.x[:, 0], self.x[:, 1], c=self.a, marker=".")
        plt.plot(self.c[:, 0], self.c[:, 1], 'x', color="black", markersize=15)
        plt.draw()
        if keep:
            plt.ioff()
        plt.show()
