from simulator import MultiServerSimulator


if __name__ == '__main__':
    simulator = MultiServerSimulator(
        n_servers=3,
        queue_size=100,
        service_time='exp',
        inter_arrival_hp_lambda=0.2,
        inter_arrival_lp_lambda=0.2,
        endtime=10000,
        seed=42
    )
    simulator.execute()
