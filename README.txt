Real-time 3-D density inversion of gravity data using deep learning

1.CNN_gravity_inversion
#Requrie tensorflow2.0(GPU version) and python3.6, need python plt library for ploting
#In order to allow the program to run directly, you need to specify all the paths in the file correctly (currently my own server path)
1). On Linux command line, after entering the directory, run "python main_predict64.py" to perform inversion imaging of gravity anomalies. (You need to change all file paths, including the weight saving location, file location, etc.)
2). On linux command line, after entering the directory, run "python 3Dinversion64.py" can demonstrate the process of training neural networks.

2.forward_modeling_lib
#Forward modeling program using C++ code(CPU version), run "./build" to generate the dynamic lib, but it requires open source libraries eigen, which can be downloaded for free on the Internet
#If you dont want to generate it by yourself, it's OK, we have put the generated dynamic library under the test_data folder
3.test_data
#Here we have saved the trained parameters and a set of test data&output 


Yidan Ding: dingyd18@mails.jlu.edu.cn
Guoqing Ma: maguoqing@jlu.edu.cn
Haoran Wang: hrw17@mails.jlu.edu.cn
College of Geo-Exploration Science and Technology, Jilin University, Changchun, 130000, China
