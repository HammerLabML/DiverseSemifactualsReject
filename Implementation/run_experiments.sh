mkdir results/

python experiments.py wine knn > results/wine_knn.txt
python experiments.py wine gnb > results/wine_gnb.txt
python experiments.py wine randomforest > results/wine_randomforest.txt

python experiments.py breastcancer knn > results/breastcancer_knn.txt
python experiments.py breastcancer gnb > results/breastcancer_gnb.txt
python experiments.py breastcancer randomforest > results/breastcancer_randomforest.txt

python experiments.py flip knn > results/flip_knn.txt
python experiments.py flip gnb > results/flip_gnb.txt
python experiments.py flip randomforest > results/flip_randomforest.txt

python experiments.py t21 knn > results/t21_knn.txt
python experiments.py t21 gnb > results/t21_gnb.txt
python experiments.py t21 randomforest > results/t21_randomforest.txt
