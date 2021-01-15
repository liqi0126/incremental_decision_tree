cd ..
python main.py --dataset $1 --tree v --exp final
python main.py --dataset $1 --tree e --exp final
for ((i=4096;i<4106;i++))
do
    python main.py --dataset $1 --tree v --shuffle --seed ${i} --exp final
    python main.py --dataset $1 --tree e --shuffle --seed ${i} --exp final
done
cd experiments

python generate_figure.py --dataset $1