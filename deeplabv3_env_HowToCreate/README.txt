Commands to create the environment 'deeplabv3_env':
1. Create the conda environment:
   conda create -n deeplabv3_env python=3.8
2. Activate the conda environment:
   conda activate deeplabv3_env
3. Install conda packages:
   conda env update --file deeplabv3_env_environment.yml --prune
4. Install pip packages:
   pip install -r deeplabv3_env_requirements.txt
