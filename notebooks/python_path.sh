# get parent directory 
PARENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PARENT_DIR/../scripts:$PYTHONPATH
