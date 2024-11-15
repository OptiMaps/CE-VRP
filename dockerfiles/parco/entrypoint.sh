#! /bin/bash

# restart ssh service
/bin/bash -c "sudo service ssh start > /dev/null"

/bin/bash -c "git clone https://github.com/OptiMaps/TrainRepo $HOME/workspace/optimap/TrainRepo"

/bin/bash -c "mv -f TrainRepo/* ."

/bin/bash -c "rm -rf TrainRepo"

/bin/bash -c "cd parco && pip install -e ."

/bin/bash
