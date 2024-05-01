#!/usr/bin/env bash

python -m trip.tools.torsion_benchmark --gpu 0 --start 0 --stop 110 --model_file /results/trip2_vanilla.pth &\
python -m trip.tools.torsion_benchmark --gpu 1 --start 110 --stop 220 --model_file /results/trip2_vanilla.pth &\   
python -m trip.tools.torsion_benchmark --gpu 2 --start 220 --stop 330 --model_file /results/trip2_vanilla.pth &\   
python -m trip.tools.torsion_benchmark --gpu 3 --start 330 --stop 439 --model_file /results/trip2_vanilla.pth &\   
python -m trip.tools.torsion_benchmark --gpu 4 --start 439 --stop 548 --model_file /results/trip2_vanilla.pth &\   
python -m trip.tools.torsion_benchmark --gpu 5 --start 548 --stop 657 --model_file /results/trip2_vanilla.pth &\
python -m trip.tools.torsion_benchmark --gpu 6 --start 657 --stop 766 --model_file /results/trip2_vanilla.pth &\
python -m trip.tools.torsion_benchmark --gpu 7 --start 766 --stop 875 --model_file /results/trip2_vanilla.pth
