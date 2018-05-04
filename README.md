# digitLearning - Alexandre Poupeau
Using the MNIST database from Yann LeCun, elaborate a neural network model in
order to identify digits made by hand.

# HOW TO
1) Create your own neural network model (OPTIONAL STEP):
  In the directory networks/model: you can create a network.
  You have to name it {network_name}.txt.
  The first line is the number of layer in the neural network (except
  the first one and the last one).
  For each layer (the number in the first line) return to a new line and
  write a number. This indicates the number of neurons in this layer.
  N.B: you are not obliged to create a new .txt file, you can simply
  use existing ones.

2) Initialize your neural network:
  Simply run the following command in the main directory:
  ./main.py networks/model/{network_name}.txt -ls 1 -ts 1 -init=S networks/saved/{dir_name}

3) Train and save:
  In order to train your neural network, run the following command:
  ./main.py networks/saved/{dir_name} -ls NB -ts NB -S
  You can choose to not save the training by deleting the -S argument.

4) Follow the evolution of your training:
  Just open the file networks/saved/{dir_name}/info.csv.
  It contains all the information about the evolution of your neural network.

5) Enjoy !
  In case of emergency do not hesitate to run the command ./main.py
  using one of the following argument:
  ["help", "-help", "--help", "h", "-h", "--h", "HELP", "-HELP", "--HELP"
   "H", "-H", "--H", "MORE", "-MORE", "--MORE", "more", "-more", "--more"]
  This will give you some indications for how to use all the arguments.

# Documentation
For further information and to get a better understanding of the code I
recommend you to read either doc/html/index.html or doc/latex/refman.pdf.
