The file `en.subject.pdf` describes the task

Tests:
- run `pytest` command in the root directory

Steps:
- `python -m venv myvenv`
- `python -m pip install -r requirements.txt`

2D:
- `python fit.py test_files/data_2d.csv`
- `python predict.py -i -f params.json` / `python predict.py -p test_files/test_2d.csv -f params.json`
- `python plot_graph.py test_files/data_2d.csv -d 2`

3D:
- `python fit.py test_files/data_3d.csv`
- `python predict.py -i -f params.json` / `python predict.py -p test_files/test_3d.csv -f params.json`
- `python plot_graph.py test_files/data_3d.csv -d 3`

Usages:
- `usage: fit.py [-h] [-v] [-s SEPARATOR] [-f FILE_WITH_PARAMS] [-n NUMBER_OF_TARGET_VALUE] path`
- `usage: predict.py [-h] (-p PATH | -i) [-s SEPARATOR] [-f FILE_WITH_PARAMS] [-v]`
- `usage: plot_graph.py [-h] -d {2,3} [-v] [-s SEPARATOR] [-f FILE_WITH_PARAMS] [-n NUMBER_OF_TARGET_VALUE] path`

Pep8:
- `pycodestyle *.py --ignore=E501`
