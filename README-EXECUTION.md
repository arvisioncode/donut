
## SET-UP

ENTORNO PREPARADO PARA LINUX:
git clone https://github.com/clovaai/donut.git
cd donut/
conda create -n donut_official python=3.7
conda activate donut_official
pip install .

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install tensorboardX
pip install pytorch-lightning==1.6.4
pip install transformers==4.11.3
pip install timm==0.5.4


<!-- WINDOWS, NECESARIO REINSTALAR ALGUNOS PAAQUETES:
    pip install pytorch-lightning==1.6.4
    pip install sconf
    pip install timm==0.5.4
    pip install zss
    pip install datasets
    pip install transformers==4.11.3
    pip install sentencepiece
    pip install tensorboard
    pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

    WINDOWS: (añadiendola en variables de entorno)
        PL_TORCH_DISTRIBUTED_BACKEND=gloo -->

huggingface-cli login
    token: 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


Para generar el dataset de train 2 opcions:
- con el NB a partir de un excell
- las anotaciones de BUPA a partir de un json
  - corregir las preguntas:
    - "Which is the result of the gene \"cMET\"?"
    - "Which is the result of the gene cMET?"
  - primero usar dataset/adapt_json_lables para generar el GT, añadir file_name, transformar los json to jsonl
  - dividir los datos entre train y test dentro de la carpeta de dataset, y con un metadata.json en cada una de ellas




## TRAINING
El daatset se genera con el NB que hice a partir de un archivo excell con anotaciones y de las imagenes
dataset have to be divided in 'test' and 'train'. Check notebooks folder

# cord
python train.py --config config/train_cord.yaml --pretrained_model_name_or_path "naver-clova-ix/donut-base" --dataset_name_or_paths '["naver-clova-ix/cord-v2"]' --exp_version "test_experiment" 

python train.py --config config/train_cord_gpu.yaml --pretrained_model_name_or_path "naver-clova-ix/donut-base" --dataset_name_or_paths '["arvisioncode/donut-funsd"]' --exp_version "donut-funsd-gpu" 



# docvqa
<!-- python train.py --config config/train_docvqa.yaml --pretrained_model_name_or_path "naver-clova-ix/donut-base-finetuned-docvqa" --dataset_name_or_paths '["nielsr/docvqa_1200_examples_donut"]' --exp_version "donut-docvqa-ft-nielsrdocvqa"  -->
python train.py --config config/train_docvqa_gpu.yaml --pretrained_model_name_or_path "naver-clova-ix/donut-base-finetuned-docvqa" --dataset_name_or_paths '["nielsr/docvqa_1200_examples_donut"]' --exp_version "donut-docvqa-ft-nielsrdocvqa" 

python train.py --config config/train_docvqa_gpu.yaml --pretrained_model_name_or_path "naver-clova-ix/donut-base-finetuned-docvqa" --dataset_name_or_paths '["aymane/donut-docvqa-oculist-test"]' --exp_version "donut-docvqa-ft-oculist" 

python train.py --config config/train_docvqa_gpu.yaml --pretrained_model_name_or_path "naver-clova-ix/donut-base-finetuned-docvqa" --dataset_name_or_paths '["./dataset/bupa_docvqa_dataset_v2/"]' --exp_version "donut-docvqa-ft-oculist" 

python train.py --config config/train_docvqa_gpu.yaml --pretrained_model_name_or_path "naver-clova-ix/donut-base" --dataset_name_or_paths '["./../rrc_docvqa/"]' --exp_version "donut-docvqa-base-rrc-ft"

## INFERENCE - EVALUATION

# FINE-TUNED::: donut-docvqa-oculist-test
python test.py --dataset_name_or_path aymane/donut-docvqa-oculist-onlytest --pretrained_model_name_or_path ./result/train_docvqa_tests/donut-docvqa-ft-oculist --save_path ./result/docvqa-output-ft.json --task_name docvqa

python test.py --dataset_name_or_path "./dataset/bupa_docvqa_part1_test/" --pretrained_model_name_or_path ./result/train_docvqa_gpu/donut-docvqa-ft-bupa-part1 --save_path ./result/donut-docvqa-ft-bupa-part1.json --task_name docvqa


# ORIGINAL::: naver-clova-ix/donut-base-finetuned-docvqa
python test.py --dataset_name_or_path aymane/donut-docvqa-oculist-onlytest --pretrained_model_name_or_path naver-clova-ix/donut-base-finetuned-docvqa --save_path ./result/docvqa-output-original.json --task_name docvqa

python test.py --config config/train_docvqa_tests.yaml --dataset_name_or_path aymane/donut-docvqa-oculist-onlytest --pretrained_model_name_or_path naver-clova-ix/donut-base-finetuned-docvqa --save_path ./result/docvqa-output-original.json --task_name docvqa

python test.py --dataset_name_or_path "./dataset/bupa_docvqa_dataset_v2/" --pretrained_model_name_or_path naver-clova-ix/donut-base-finetuned-docvqa --save_path ./result/donut-docvqa-ft-bupa-part1and2-original.json --task_name docvqa


python test.py --dataset_name_or_path aymanechilah/donut-docvqa-concert1 --pretrained_model_name_or_path naver-clova-ix/donut-base-finetuned-docvqa --save_path ./result/donut-docvqa-ft-bupa-part1and2-original.json --task_name docvqa


## SINGLE INFERENCE
I make an script for docvqa: inference.py

python inference.py --pretrained_model_name_or_path naver-clova-ix/donut-base-finetuned-docvqa --image_path ./dataset/bupa_docvqa_dataset_v2/train/Caris_12345678A_05.jpg --question "What is the page number?"

python inference.py --pretrained_model_name_or_path naver-clova-ix/donut-base-finetuned-docvqa --image_path ./dataset/image_divided/Caris_12345678B_01.jpg --question "What is sex?"

## RESULTS 

NECESITA MUCHAS EPOCH ARA A`RENDER BIEN!!! CON 300 MALOS RESUTLADOS CON 2000 BUENOS
EL PROBLEMA PUEDE SER OVERFITTING!!!!!!!!!!
TODO: CHECAK RESULTADOS EN EL DATASET ORIGINAL , PARA VER SI ESTE MODELO TUNEADO EMPEORA AHI



## ONNX
Para configurar el DONUT2ONNX, tengoq ue usar una maquina remota
Al ser remota, instalo jupyter en un entorno de conda, y luego sigo estos pasos para poder usar jupyter en remoto:
https://docs.anaconda.com/anaconda/user-guide/tasks/remote-jupyter-notebook/ 


## FINALMENTE SI SE PUDO ENTRENAR CON EL NOTEBOOK, USAR EL ORIGINAL PARA PROBAR Y LUEGO CAMBIAR LOS DATOS