import math
import matplotlib.pyplot as plt
import numpy as np

def hessian_r(C,w,g):
  n=w.size
  d=g.dot(w)
  if math.fabs(d)<1e-5:
    return np.identity(n)
  A=np.zeros((n,n))
  B=np.zeros((n,n))
  for i in range(n):
    A[i]=w[i]*g
    B[i]=w[i]*w
  A /=d
  At=A.transpose()
  B /=d
  I=np.indentity(n)
  return (I-A).dot(C).dot(I-At)+B
  
  def hessian_r2(C,w,g): #DFP算法 w,g是△w,△g
  n=w.size
  d=g.dot(w)
  if math.fabs(d)<1e-4:
    return np.identity(n)
  A=np.zeros((n,n))
  for i in range(n):
    A[i]=g[i]*g
  A /= d

  B=np.zeros((n,n))
  w2=C.dot(w)
  for i in range(n):
    B[i]=w2[i]*w2
  d2=w.dot(w2)
  if math.fabs(d2)<1e-4:
    return np.identity(n)
  B /=d2
  return C+A-B
  
  def lr_BFGS(data,alpha):
    n=len(data[0])-1
    w0=np.zeros(n)  #上一个权重
    w=np.zeros(n)  #当前权值
    g0=np.zeros(n) #上一个梯度
    g=np.zeros(n)  #Hession矩阵的逆矩阵，初始化为单位(n*1)
    for times in range(1000):
      for d in data:
        x=np.array(d[:-1]) #输入数据(n维列向量)
        y=d[-1]
        C=hessian_r(C,w-w0,g-g0)
        g0=g
        w0=w
        g=(y-pred(w,x))*x
        w=w+alpha*C.dot(g)
      print (times,w)
    return w
