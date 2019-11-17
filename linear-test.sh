docker run -v $PWD:/app -w /app -it els-python python ./linear.py -i airfoil_self_noise.dat -s '\t' -r
docker run -v $PWD:/app -w /app -it els-python python ./linear.py -i winequality-red.csv -s ';' -e 1 -r
docker run -v $PWD:/app -w /app -it els-python python ./linear.py -i winequality-white.csv -s ';' -e 1 -r
