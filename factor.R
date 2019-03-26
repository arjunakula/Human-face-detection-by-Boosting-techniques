n = 10
p = 5
d = 2
sigma = 1.
IT = 1000

W_true = matrix(rnorm(d*p), nrow=p)
Z_true = matrix(rnorm(n*d), nrow=d)
epsilon = matrix(rnorm(p*n)*sigma, nrow=p)
X = W_true%*%Z_true + epsilon

sq = 1.;
XX = X%*%t(X)
W = matrix(rnorm(p*d)*.1, nrow=p)
for (it in 1:IT)
{
  A = rbind(cbind(t(W)%*%W/sq+diag(d), t(W)/sq), cbind(W/sq, diag(p)))
  AS = mySweep(A, d)
  alpha = AS[1:d, (d+1):(d+p)]
  D = -AS[1:d, 1:d]
  Zh = alpha %*% X
  ZZ = Zh %*% t(Zh) + D*n
  B = rbind(cbind(ZZ, Zh%*%t(X)), cbind(X%*%t(Zh), XX))
  BS = mySweep(B, d)
  W = t(BS[1:d, (d+1):(d+p)])
  sq = mean(diag(BS[(d+1):(d+p), (d+1):(d+p)]))/n;
  sq1 = mean((X-W%*%Zh)^2)
  print(cbind(sq, sq1))
}
print(W)