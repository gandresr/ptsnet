from heapq import heappush, heappop, nsmallest

q = []
heappush(q, (100, 'update S2', 2))
heappush(q, (1.5, 'update S1', 3))
heappush(q, (3, 'update S0', 4))
heappush(q, (4, 'update S4', 5))
min(q)