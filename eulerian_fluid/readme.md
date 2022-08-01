## Quantized Eulerain Fluid Simulation

Please see `python demo.py -h` for more details of how to run the smoke simulation

### Large scale demo

To reproduce the large scale somke simulation demo, please use
```
python run.py --demo [0/1] -o outputs
```
Set the arg of `demo` to `0` for the bunny demo and `1` for the flow demo. `-o outputs` means the set the output folder to `outputs`.

#### To visualize this demo
The final demo in the paper is rendered by Blender. Here we attach a fast renderer. One can type

```
python render.py -d 3 -r 768 -i outputs/np -o outputs/rendered_bunny --frames 300
```
to visualize the demo of buuny or 
```
python render.py -d 3 -r 896 -i outputs/np -o outputs/rendered_flow --frames 280
```
to visualize flow demo.

### Shared exponent effectiveness experiment

We conducted an experiment on a 2D smoke simualtion to verify the effectiveness of shared exponent.

Please use the following command to run the experiment :
```
python run_shared_exp.py --data-type [0/1/2/3] -o outputs
```
data type of `dey density`: 
+ 0 for float32
+ 1 for shared exponent 
+ 2 for fixed-point
+ 3 for non-shared exponent 

### Performance comparisons
Please use the following command to run the experiment :
```
python run_benchmark.py -e [0/1/2]
```
`-e` means experiment
+ 0 for all float32
+ 1 for shared_exponent velocity(5 exp + 9 fractions fixed 10 for dy density
+ 2 for all fixed 10
