FROM mariuswi/trajopt_ws:1.0
RUN wget http://pylegacy.org/hub/get-pip-pyopenssl.py
RUN python get-pip-pyopenssl.py
RUN pip install pybullet --trusted-host pypi.org --trusted-host files.pythonhosted.org
WORKDIR /root/catkin_ws/src
