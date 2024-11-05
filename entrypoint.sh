#! /bin/bash

# restart ssh service
/bin/bash -c "sudo service ssh start > /dev/null"

/bin/bash -c "git clone https://github.com/OptiMaps/TrainRepo $HOME/workspace/graph-mamba/TrainRepo"

/bin/bash -c "mv -f TrainRepo/* ."

/bin/bash -c "rm -rf TrainRepo"

/bin/bash

# execute jupyter lab server
# /bin/bash -c "cd $HOME/workspace/graph-mamba && \
#     poetry run jupyter lab --ip 0.0.0.0 --allow-root \
#     --NotebookApp.token= --no-browser --notebook-dir=$HOME"
