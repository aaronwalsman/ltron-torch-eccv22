

def train_uct(train_config):
    
    print('='*80)
    log = SummaryWriter()
    clock = [0]
    
    model = build_reassembly_model(train_config)
    optimizer = build_optimizer(train_config, model)
    train_env = build_train_env(train_config)
    
    for epoch in range(1, train_config.epochs+1):
        epoch_start = time.time()
        print('='*80)
        print('Epoch: %i'%epoch)
        
        episodes = rollout_uct_epoch(
            train_config, train_env, model, 'train', log, clock)

def rollout_uct_epoch(train_config, train_env, model, log, clock):
    print('-'*80)
    print('Rolling out episodes')
    
    model.eval()
    
    with torch.no_grad():
        for t in tqdm.tqdm(range(train_config.train_trees_per_epoch))
            '''
            # statistics
            N(s) = state visit count
            N(s,a) = state/action visit count
            W(s,a) = total action value ?
            Q(s,a) = mean action value
            
            # prior (I think this is learned)
            P(s,a) = prior probability of taking a in s (actor network output)
            
            # exploration rate
            C(s) = log((1 + N(s) + c_base)/c_base) + c_init
            
            U(s,a) = C(s) P(s,a) (N(s))^0.5 / (1+N(s,a))
            a_t = argmax_a (Q(s_t, a) + U(s_t, a))
            
            in my case h = history (because I don't have states)
            '''
            
            observation = train_env.reset()
            o0 = hashable_observation(observation)
            s0 = env.get_state()
            
            cached_obs = {s0:observation}
            
            N = {():0}
            W = {}
            Q = {}
            
            for r in range(train_config.train_steps_per_tree):
                # ok, here's how this is going to work with our current
                # parallel model:
                # 1. Everything is just constantly running asynchronously
                # 2. We cache all the action probabilities, so we only run
                #    forward passes for parts of the rollout we've never seen
                #    before.
                # 3. Sampling what to do next when we have cached action
                #    probabilities is just math.  Costs very little.
                # 4. So for every step, for anything that is on-tree, we run it
                #    forward until all batches items are off-tree.
                # 5. If anything finishes an entire episode on-tree, then we
                #    just update the statistics and reset that one element, and
                #    run again until we hit something off-tree.
                # 6. Once everything is off tree, we run a single
                #    forward pass and continue to the next iteration.
                # 6.A. This is wasteful though because we have to rerun with the
                #    whole history every time, and can't make use of the
                #    memory caching.
                # 6.B. What we can do instead is every time we hit a leaf, we
                #    run each episode out the end.  The issue here would be
                #    that this rollout will have different lengths for each
                #    episode in the batch.  But our caching/memory model in the
                #    transformer is already built to handle that.  So when
                #    one thing finishes, we just run that one thing until we're
                #    off tree again, then resume running everything.
                # X. Also, we can use a single integer to represent a decision,
                #    and a sequence of such integers for a full history for
                #    N, W, Q, etc. if we go Klemen's model where we can only do
                #    one thing at a time and just update some internal simulator
                #    state.
                if h in N:
                    # we already have p_h somewhere to look up, look it up
                else:
                    # time to start doing forward passes
                p_h = model(h)
                u_sa = Q[h,a] + c_puct * p_ha * (N[h]**0.5) / (1 + N[h,a])
                a = argmax(u_sa)
