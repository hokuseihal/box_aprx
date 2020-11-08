for num_model in 1 2 3 4;do
  for num_layer in 3 6 12 24 36 48 60;do
    for feature in 128 256 512 1024 2048;do
      echo p train.py --num_model $num_model --num_layer $num_layer --feature $feature;
      python3 train.py --num_model $num_model --num_layer $num_layer --feature $feature;
    done
  done
done
