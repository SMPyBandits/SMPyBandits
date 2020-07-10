from matplotlib import pyplot as plt
from SMPyBandits.Policies import wSWA, FEWA, EFF_FEWA, RAWUCB, EFF_RAWUCB
import os
import numpy as np


def delay_data(T, m=2):
  res= []
  sigma = 1
  policy = EFF_FEWA(1, subgaussian=sigma, alpha=0.06, delay=True, m=m)
  for t in range(T):
    policy.getReward(0, 1)
    res.append(policy.delay.copy())
  res[-1] = res[-1][~np.isnan(res[-1])]
  final_length = len(res[-1])
  for i, r in enumerate(res):
    res[i] = np.append(r[:final_length], np.nan * np.ones(final_length - len(r[:final_length])))
  return np.arange(1, T+2 ,1), policy.windows[:final_length], np.array(res) / policy.windows[:final_length]

def delay_plot(x,w,Z,T,m):
  fig, ax = plt.subplots(figsize=(12, 7))
  ax.set_xticks(np.arange(len(x)), minor=True)
  ax.set_yticks(np.arange(len(w)), minor=True)
  ax.set_xlabel("$N_i$")
  ax.set_ylabel("$j$", rotation=0, labelpad = 18)
  title = "$m=%s$"%m
  ax.set_title(title, pad = 12)
  y = np.arange(0, len(w)+1,1)
  X,Y = np.meshgrid(x,y)
  c = ax.pcolormesh(X-0.5, Y-0.5, Z.transpose(), cmap='RdYlGn_r', vmin= 0, vmax=1, rasterized =True)
  fig.colorbar(c, ax=ax)
  plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
  name = 'T=%s_m=%s' % (T, m)
  plt.savefig(name + '.pdf', dpi=1200)

def compute_h_omega(h_max = 10**15, m=2):
  h=[1]
  omega = [1]
  j=0
  while h[-1] < h_max:
    h.append(np.int64(np.ceil(m*h[-1])))
    omega.append(np.int64(omega[-1]*(1+ np.floor((h[-1]-h[-2]-1)/omega[-1]))))
    j+=1
  return np.array(h, dtype=np.int), np.array(omega, dtype=np.int)

def plot_h_omega():
  M = np.random.rand(100)+1
  M = np.append(M, 2)
  min = []
  max = []
  median = []
  mean = []
  for m in M:
    h, omega = compute_h_omega(m=m)
    filter = np.where(omega!=1)
    ratio = omega[filter]* m /(m-1)/h[filter]
    min.append(np.min(ratio))
    max.append(np.max(ratio))
    median.append(np.quantile(ratio,0.5))
    mean.append(np.mean(ratio))
  plt.plot(M, min, 'kv', label='min', markersize=4, linewidth=1)
  plt.plot(M, max, 'k^',label='max', markersize=4, linewidth=1 )
  plt.plot(M, mean, 'ko', label='mean', markersize=4, linewidth=1)
  plt.plot(M, median, 'k+', label ='median', markersize=4, linewidth=1)
  plt.legend()
  plt.xlabel('$m$')
  plt.ylabel(r'$\frac{m \omega_j}{(m-1)h_j}$', rotation=0, labelpad = 18)
  plt.savefig('delay_ratio.pdf')





if __name__ == "__main__":
  plot_h_omega()