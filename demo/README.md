Activate the conda enviranment with torch installed.

## Coordinator

Coordinator port can be changed in config.py

Run coordinator as:
```bash
python coordinator.py
```
If you get port in use error, you can kill the process as (replace <port_no>)
```bash
kill $(lsof -ti:<port_no>)
```

## Training Jobs

Run the job as, with a job name (here job1) and number of training steps to run (here 2000):
```bash
python train.py job1 2000
```

## Reconfiguring

To submit a reconfig request, on the coordinator terminal enter:
```bash
job_name-dev1,dev2
```
(dev1 and dev2 can be same)

Example, to move job1 to devices 2 and 3
```
job1-2,3
```
Curently the reconfig logic simply moves the first half of model to dev1 and second half to dev2

## Watching GPU's
To observe GPU's through the process, start watching GPU stats in a terminal as:
```bash
watch -n 0.5 nvidia-smi
```

If ever some process doesn't stop or if GPU memory needs to be cleared, you can use:
```bash
nvidia-smi | awk '$2=="Processes:" {p=1} p' | awk '$5>0 {print $5}' | xargs kill -9
```


## License
[MIT](https://choosealicense.com/licenses/mit/)
Replace this with UIUC one