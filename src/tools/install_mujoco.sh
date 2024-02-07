wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
mv mujoco210 ~/.mujoco/
rm -r mujoco210-linux-x86_64.tar.gz
pip install "cython<3" gymnasium~=0.29.1 mujoco-py~=2.1.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
