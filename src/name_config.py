import numpy as np
import os

def generate_run_name_and_directories(debug=False):
    WEATHER_CONDITIONS = [
        'sunny', 'cloudy', 'rainy', 'snowy', 'windy', 'stormy', 'foggy', 'hail', 
        'thunderstorm', 'tornado', 'hurricane', 'blizzard', 'drizzle', 'sleet', 
        'dust storm'
    ]
    COLOURS = [
        'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 
        'pink', 'brown', 'grey', 'violet', 'indigo', 'turquoise', 'gold'
    ]

    RUN_NAME = f'{WEATHER_CONDITIONS[np.random.randint(0,14)]}-{COLOURS[np.random.randint(0,14)]}'

    if not debug:
        if not os.path.exists('./runs/ProGAN/checkpoints'):
            print('Creating Checkpoint Directory')
            os.makedirs('./runs/ProGAN/checkpoints')

    if not debug:
        os.makedirs(f'./data/runs/ProGAN/{RUN_NAME}/log', exist_ok=True)
        os.makedirs(f'./data/runs/ProGAN/{RUN_NAME}/fake', exist_ok=True)
        os.makedirs(f'./data/runs/ProGAN/{RUN_NAME}/real', exist_ok=True)

    return RUN_NAME


if __name__ == '__main__':
    RUN_NAME = generate_run_name_and_directories(debug=True)
    print(RUN_NAME)



