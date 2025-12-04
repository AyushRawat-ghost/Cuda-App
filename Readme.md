<!-- Building the file -->
docker-compose up -d --build

<!-- container Attachment -->
docker exec -it cuda_cpp_dev_env /bin/bash

<!-- GPU Access  -->
docker run -it --rm --gpus all cuda_cpp_dev_env 

<!-- Version  -->
nvcc --version

<!-- pytorch verification  -->
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device name: {torch.cuda.get_device_name(0)}') "