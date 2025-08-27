dir=plots/$1

mkdir -p $dir

samples=1000
start="2004-05-16"
end="2004-05-22"
filenames=single_$samples
ensembesize=30
inputfile=Beijing.nc
#inputfile=Abisko.nc

rm -f *.json
python3 main.py --file $inputfile --start $start -e $end -g new \
   -n "$dir/compo" --samples $samples --ensemble 1 --close_figs --basefig

for n in $(seq -w 001 $ensembesize ) ; 

do 
  echo ------------- Ensemble member $n
  rm -f *.json
  if [[ "$2" -eq "1" ]] 
    then
      python3 main.py --file $inputfile --start $start -e $end -g new \
            -n "$dir/${filenames}_${n}" --samples $samples --ensemble 1 --close_figs
  fi

  composite -compose Multiply "$dir/${filenames}_${n}.png" $dir/compo.png $dir/compo.png

done
