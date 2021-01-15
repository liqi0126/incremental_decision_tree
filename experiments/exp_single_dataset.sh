python main.py --dataset $1 --tree v --unshuffle --seed ${i + 4096} --exp final
python main.py --dataset $1 --tree e --unshuffle --seed ${i + 4096} --exp final
for ((int i=0;i<10;i++))
    python main.py --dataset $1 --tree v --shuffle --seed ${i + 4096} --exp final
    python main.py --dataset $1 --tree e --shuffle --seed ${i + 4096} --exp final

python generate_figure.py --dataset $1