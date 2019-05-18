FROM ubuntu:18.04
RUN apt-get update && apt-get install -y openssh-server mc mpich
#Create mpiuser
RUN useradd -ms /bin/bash mpiuser
RUN echo 'mpiuser:mpi123' | chpasswd
#setup ssh
RUN mkdir /var/run/sshd
RUN echo 'root:s123456' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
COPY ./cloud /home/mpiuser/cloud
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
