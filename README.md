## Simplified network model of CA3 featuring sequence replay and ripple oscillation

To run:

    git clone https://github.com/KaliLab/ca3net.git
    pip install -r requirements.txt
    cd ca3net
    mkdir figures
    cd scripts
    python generate_spike_train  # generate CA3 like spike trains (as exploration of a maze)
    python stdp.py  # learns the recurrent weight (via STDP, based on the spiketrain)
    python spw_network.py  # creates the network, runs the simulation, analyses and plots results

![](https://raw.githubusercontent.com/KaliLab/ca3net/master/summary.png)

> We haven't published this yet, so please don't do before us...
