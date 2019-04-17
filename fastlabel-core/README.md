# fastlabel-core

#### 介绍
fastlabel-core服务器端

#### 软件架构
软件架构说明


#### 安装教程

（Base on Centos6）
1.系统更新
0x0 
    sed -i "s/SELINUX\=enforcing/SELINUX\=disabled/g" /etc/sysconfig/selinux
    setenforce 0
    yum install epel-release.noarch  -y
    yum update -y
    reboot
2.安装系统依赖和python3
  yum install gcc gcc-c++ zlib-devel openssl-devel libpng-devel libjpeg-devel libxml2-devel cmake bzip2-devel ncurses-devel sqlite-devel readline-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel -y
  wget https://www.python.org/ftp/python/3.7.2/Python-3.7.2.tgz
  tar xvf Python-3.7.2.tgz
  cd Python-3.7.2
  ./configure --enable-optimizations
  make&&make install

3.安装monogodb
  vi /etc/yum.repos.d/mongodb.repo 写入如下内容

  [MongoDB]
name=MongoDB Repository
baseurl=http://repo.mongodb.org/yum/redhat/$releasever/mongodb-org/4.0/x86_64/
gpgcheck=0
enabled=1
保存后安装mongodb4
yum install mongodb-org
service mongod start

4.安装依赖
  git clone https://gitee.com/cellsvision/fastlabel-core.git
  cd fastlabel-core
  pip3 install -r requirements.txt
  pip3 install gunicorn==19.9.0
  pip3 install -r requirements.txt
  pip3 install pycocotools
5.修改配置启动服务
  修改config.py 里面的env环境变量，特别是mongodb连接
  gunicorn -w 1 -b 0.0.0.0:5000 app:app --worker-class eventlet --log-level debug --no-sendfile



