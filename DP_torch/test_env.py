from packing.scenario import Scenario
from packing.cell.cell_gym import CellEnv

if __name__ == '__main__':
    scenario = Scenario()
    # create world
    packing = scenario.build_packing()
    # create cell environment
    agent_env = CellEnv(packing, scenario.reset_packing, scenario.reward, scenario.observation, method="rotation")
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env,i) for i in range(env.n)]
    # execution loop
    obs_n = agent_env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
        env.render()
        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))