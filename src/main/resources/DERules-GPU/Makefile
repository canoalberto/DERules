all: javah gpu library
javah: 
	javac -classpath ../../../../target/DERules-1.0-jar-with-dependencies.jar -h jni ../../java/moa/classifiers/evolutionary/DERules.java
gpu: DERules.cu
	nvcc -O3 -Xcompiler -fPIC -shared -c DERules.cu -o DERules.o -I /usr/local/cuda/include -I /usr/lib/jvm/default-java/include -I /usr/lib/jvm/default-java/include/linux -L /usr/local/cuda/lib64 -lcuda -lcudart
library: DERules.o
	g++ -O3 -fPIC -shared -Wl,-soname,DERules DERules.o -o libDERules.so -L /usr/local/cuda/lib64 -lcuda -lcudart
clean:
	rm DERules.o -o libDERules.so