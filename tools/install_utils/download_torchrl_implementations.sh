git clone --depth 1 --filter=blob:none --sparse https://github.com/pytorch/rl.git
cd rl
git sparse-checkout set sota-implementations
mv sota-implementations/ ../torch_rl_implementations
cd ..
rm -rf rl
touch torch_rl_implementations/.gitignore
echo "*" > torch_rl_implementations/.gitignore
