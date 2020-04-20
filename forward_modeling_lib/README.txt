#build the lib with the following command
#generate the dynamic lib of gravity forward for python, this is a CPU version
#this program require eigen openlib,please download for free on the Internet
g++ -I./lib/eigen/ -o libpyforward.so -shared -fPIC gravity_forward_python.cpp 
