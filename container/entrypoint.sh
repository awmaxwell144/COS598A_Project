#!/bin/bash
if [ -n "$SINGULARITY_NAME" ]; then
    echo "Inside a Singularity container, so not running fixuid."
else
    echo "Inside a Docker container, running fixuid to enable writable binds."
    fixuid
fi
unset FPATH
exec "$@"
