# CameraTraps
preparar VM: from vm
virtualenv -p python3 .venv && source ./.venv/bin/activate
pip install notebook

copiar carpetas: from local
rsync --inplace -av -e 'ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa_cw' /Users/jcuomo/CamaraTrampa jcuomo@10.146.175.65:/home/jcuomo

reemplazar full paths: vm
find . -type f -exec sed -i 's|content/drive/MyDrive/CamarasAspen|home/jcuomo/CamaraTrampa|g’ {} +

find . -type f -exec sed -i ‘s|content/drive/MyDrive/CamarasAspen|home/ubuntu/CamaraTrampa|g’ {} +

correr jupyter: vm
jupyter notebook --no-browser --port=8888


levantar jupyter: local
ssh -N -L localhost:8888:localhost:8888 -i /Users/jcuomo/.ssh/id_rsa_cw jcuomo@10.145.36.211
abrir en el navegador http://localhost:8888/

agregar al JN:
!pip install torchvision
!pip install Pillow
!pip install torch
!pip install transformers



nohup python your_training_script.py > output.log &


copy from external disk to remote server showing progress
rsync --inplace -a  --info=progress2 -e 'ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa_cw' "/Volumes/Samsung_T5/CamaraTrampa/Processed Pics Ecology ms/" jcuomo@10.144.214.207:/home/jcuomo/CamaraTrampa/labeled