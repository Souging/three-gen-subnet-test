
install pm2
```commandline
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash && source /root/.bashrc && nvm install node && npm install pm2@latest -g
```


   
-------
run  Models(Choose one)
-------
neurons/miner/workers.py corresponds to img_to_3d

neurons/miner/workers_test.py corresponds to text_to_3d

1.|-----image_to_3D

# Clone repository
```commandline
git clone https://huggingface.co/spaces/cavargas10/TRELLIS-TextoImagen3D
cd TRELLIS-TextoImagen3D
```
# Create and activate Python environment
```commandline
python -m venv env
source env/bin/activate
```
# Install dependencies and run
```commandline
pip install -r requirements.txt
pip install openai
pip install --upgrade gradio gradio_client


pm2 start --name app "CUDA_VISIBLE_DEVICES=0  python app.py --port 10000 --model img"
```

2.|----text_to_3d
```commandline
git clone https://huggingface.co/spaces/souging/TRELLIS_TextTo3D

cd TRELLIS_TextTo3D
python -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install openai

pm2 start --name model1 "CUDA_VISIBLE_DEVICES=0  python app.py --port 10000"

CUDA_VISIBLE_DEVICES=0 python app.py --port 20000

/root/miniconda3/envs/three-gen-neurons/bin/python -m pip install gradio_client
```


-------
run val
--------
install miniconda

```commandline
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_25.1.1-2-Linux-x86_64.sh
chmod -R 777 Miniconda3-py310_25.1.1-2-Linux-x86_64.sh
./Miniconda3-py310_25.1.1-2-Linux-x86_64.sh
```

```commandline
git clone https://github.com/Souging/three-gen-subnet-test.git
cd three-gen-subnet-test/validation
./setup_env.sh
```
change port for =>   validation.config.js.bak to validation.config.js
run it
```commandline
pm2 start validation.config.js
```


-------
run miner
--------
```commandline
cd three-gen-subnet-test/neurons
./setup_env.sh
```
change  for =>   miner.config.js.bak to miner.config.js
run it
```commandline
pm2 start miner.config.js
```
