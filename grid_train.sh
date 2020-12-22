for model in 1 2 3 4;do
  echo p train.py --model $model --linux;
  python3 train.py --model $model --linux;
done
