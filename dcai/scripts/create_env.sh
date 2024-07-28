env_dir=.venv_dcai
[[ -d $env_dir ]] && rm -rf $env_dir
python3 -m venv $env_dir
source $env_dir/bin/activate
which python
pip install -r requirements.txt