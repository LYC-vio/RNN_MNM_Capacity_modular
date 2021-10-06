from training.train import train

if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='data/Capacity_stim1-4_clrwht_6tasks_16perring_512_1s05s_mx')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--neachring', type=int, default=16)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    hp = {# number of units each ring
          'n_eachring': args.neachring,
          # number of rings/modalities
          'num_ring': 3,
          'activation': 'softplus',
          'n_rnn': 512,
          'learning_rate': 0.001,
          'mix_rule': True,
          'l1_h': 0.,
          'use_separate_input': False,
          'target_perf': 0.995,
          'mature_target_perf': 0.95,
          'mid_target_perf': 0.65,
          'early_target_perf': 0.35,}

    train(args.modeldir,
        seed=args.seed,
        hp=hp,
        ruleset='Capacity_stim1-4_clrwht_6tasks_1s05s_mx',
        rule_trains=['overlap','zero_gap','gap','odr','odrd','gap500',\
            'Capacity_color_1_stims_wht_stims_1s05s_mx','Capacity_color_2_stims_wht_stims_1s05s_mx','Capacity_color_3_stims_wht_stims_1s05s_mx',\
                'Capacity_color_4_stims_wht_stims_1s05s_mx',],
        display_step=500,
        continue_after_target_reached=True,)