---
layout: default
---
# Installation 
You can install ECOC by downloading the pre-compiled binary from the  [release](https://github.com/chenhao392/ecoc/releases) or compile from source. However, please note that ECOC calls C functions from the [liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)  package and requires its dynamic libraries installed, which may not be a default in your machine. Sample installation and configuration for Ubuntu/Mac are provided below. 

##  liblinear installation
 
```
# install liblinear
tar -xf liblinear-2.30.tar.gz
cd liblinear-2.30
make lib
ln -s liblinear.so.3 liblinear.so

# config for osx
# please add the following to your ~/.bash_profile
export LD_LIBRARY_PATH="/path/to/liblinear-2.30:$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="/path/to/liblinear-2.30:$DYLD_LIBRARY_PATH"
export C_INCLUDE_PATH="/path/to/liblinear-2.30:$C_INCLUDE_PATH"

# config for ubuntu
# please add the following to your ~/.bashrc
export LD_LIBRARY_PATH="/path/to/liblinear-2.30:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/path/to/liblinear-2.30:$LIBRARY_PATH"
export C_INCLUDE_PATH="/path/to/liblinear-2.30:$C_INCLUDE_PATH"
```

## compile ECOC from source
With GO environment installed in your's machine, you can compile ECOC via `go build`. You can either download the source file from [release](https://github.com/chenhao392/ecoc/releases) or directly from Github. The following commands compile ECOC using the latest code on Github. 
```
# compile ECOC from source
git init
git pull https://github.com/chenhao392/ecoc
go build
```
# Common problem
Missing liblinear library is the most common problem. If the liblinear package is either not installed or not correctly configured, You'll see error messages similar to the following.   If so, please check your liblinear installation and ask the further question on Github.

##  ***Error for running the downloaded binary.*** 
I downloaded the pre-compiled binary file from the release. When I try to execute, it complains that liblinear.so.3 can not be found, such as the following. 
```
# error msg from Ubuntu
error while loading shared libraries: liblinear.so.3: cannot open shared object file: No such file or directory
```
```
# error msg from osx
dyld: Library not loaded: liblinear.so.3
Referenced from: /Users/chen/work/ecoc/./ecoc
Reason: image not found
Abort trap: 6
```


##  ***Error in compiling the source code.*** 
 I downloaded the source files and tried to compile it. But it complains that it cannot find -llinear or linear.h. 
```
# missing shared library
/usr/local/go/pkg/tool/linux_amd64/link: running gcc failed: exit status 1
/usr/bin/ld: cannot find -llinear
collect2: error: ld returned 1 exit status

# missing head file
src/ml_linear.go:5:10: fatal error: 'linear.h' file not found
#include <linear.h>
^~~~~~~~~~
1 error generated.
```

