wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
mkdir -p ~/.mujoco/mujoco210
cp -r mujoco210 ~/.mujoco/
rm -r mujoco210-linux-x86_64.tar.gz mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

pip install "cython<3" gymnasium[mujoco]~=0.29.1 mujoco-py~=2.1.2
