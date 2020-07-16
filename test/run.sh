#!/bin/bash

# sigma = 0.1
# eta = 1
# alpha = M+eta
#def cat_prior(N,alpha,M,K,mask=True):
#    return gammaln(alpha*K)-K*gammaln(alpha) - (1-mask)*gammaln(N + alpha*K)

for n in chomp heli glob; do
  echo "${n} = ["
  for i in 18 42 78 114 150 222 258 330 438; do
    #mkdir -p $n/$i
    #python3 ../test_bbm.py $n $i | tee $n/$i/run.log
    #echo -n "(${i}, "
    sed -n -e 's/Classification//p' $n/$i/run.log | tr -d '\n'
    echo ","
    #echo "),"
  done
  echo "],"
done

for n in chomp heli glob; do
  echo $n
  for i in 18 42 78 114 150 222 258 330 438; do
    echo -n "$i "
    python3 parser.py $n/$i/run.log
  done
done
