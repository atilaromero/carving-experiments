#!/bin/bash -e

if ! [ -e govdocs1 ]
then
        mkdir govdocs1
fi

cd govdocs1

if ! [ -e src ]
then
        mkdir src
fi

pushd src

for x in `seq -w 1 999`
do
        if ! [ -e $x ]
        then
                if ! [ -e ${x}.zip ]
                then
                        wget -c http://downloads.digitalcorpora.org/corpora/files/govdocs1/zipfiles/${x}.zip
                fi
                7z x ${x}.zip
        fi
done

popd

if ! [ -e by-type ]
then
        mkdir by-type
        find src/*/ -type f | while read i
        do
                ext=${i##*.}
                echo $ext
                mkdir -p by-type/$ext
                ln -s ../../$i by-type/$ext/
        done
fi

if ! [ -e sample200 ]
then
        mkdir sample200
        pushd sample200
        for i in  `ls ../by-type/`
        do
                mkdir $i
                ls ../by-type/$i/ | shuf -n 200 | xargs -I XXX ln -s ../../by-type/$i/XXX $i/
        done
        popd
fi

