#!/bin/bash

for file in configs/*.yaml; do
    if [ "$file" != "configs/default.yaml" ]; then
        echo "Running $file"
        python main.py --cfg="$file"
    fi
done
