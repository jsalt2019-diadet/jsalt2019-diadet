#! /bin/bash

printf "\t File Check in folder: %s.\n" "$1"
mkdir -p $1/.backup

for file in $1/*; do 
    if [ -f $file ]; then
        echo "$file exists. Moved old file to .backup"
        cp $file $1/.backup/$(basename $file)
        rm $file
        touch $file
    fi
done



