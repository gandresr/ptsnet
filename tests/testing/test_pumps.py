import numpy as np
import matplotlib.pyplot as plt

H = [90.2, 90.1, 89.8, 88, 84, 76, 67, 53]
Q = [0, 100, 200, 300, 400, 500, 600, 700]
A, B, C = np.polyfit(Q, H, 2)
print(A,B,C)

HR = H[4]
QR = Q[4]

Q = np.linspace(0, 1000, 10)

alpha = C**0.5
a2 = B/alpha
a1 = A

H = (a1*Q**2 + a2*alpha*Q + alpha**2)

