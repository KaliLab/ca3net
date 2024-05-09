## Data-driven network model of CA3 - featuring sequence replay and ripple oscillation

Code repository of the [eLife article _"Hippocampal sharp wave-ripples and the associated sequence replay emerge from structured synaptic interactions in a network model of area CA3"_](https://elifesciences.org/articles/71850)

To run:

    git clone https://github.com/KaliLab/ca3net.git
    cd ca3net
    pip3 install -r requirements.txt
    mkdir figures
    cd scripts
    python generate_spike_train.py  # generate CA3 like spike trains (as exploration of a maze)
    python stdp.py  # learns the recurrent weight (via STDP, based on the spiketrain)
    python spw_network.py  # creates the network, runs the simulation, analyses and plots results

![](https://raw.githubusercontent.com/KaliLab/ca3net/master/summary.png)
