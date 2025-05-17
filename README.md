# start_guildline_for_Pygames_and_machine_learning
start guildline for Pygames and machine learning

<p align="center" width="100%">
    <img width="25%" src="https://github.com/jkaewprateep/start_guildline_for_Pygames_and_machine_learning/blob/main/Python.jpg">
    <img width="24%" src="https://github.com/jkaewprateep/start_guildline_for_Pygames_and_machine_learning/blob/main/pygame.jpg">
    <img width="18%" src="https://github.com/jkaewprateep/start_guildline_for_Pygames_and_machine_learning/blob/main/image10.jpg">
    <img width="12%" src="https://github.com/jkaewprateep/start_guildline_for_Pygames_and_machine_learning/blob/main/image6.jpg"> </br>
    <b> Pygame and Tensorflow AI machine learning </b> </br>
    <b> ( Picture from Internet ) </b> </br>
</p>

[Flappybird games]( https://pygame-learning-environment.readthedocs.io/en/latest/user/games/flappybird.html#rewards )

🧸💬 First of all, start from a simple Python project with the Pygame library environment to create a learning environment for our machine learning. A simple project will be provided as this. </br>

```
x = 100
y = 100
import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" %(x,y)

import ple;
from ple.games.snake import Snake;
from ple import PLE;
from pygame.constants import K_w, K_a, K_d, K_s, K_h

import signal

from os.path import exists
import tensorflow as tf;
```

🐐💬 Create custom variables for the program and machine learning. The wide usage of variables needs to be updated to include global variables rare internal variables because the program examines the variable scopes, and for security, the application accesses the same global variables. </br>

```
game = Snake(width=216, height=384)               # 🐑💬 ➰ Create game play environment for our experiment
p = PLE(game, fps=30, display_screen=True)
nb_frames = 1000000000
learning_rate = 0.0001
momentum = 0.4
batch_size = 1
actions = dict({ "w": K_w, "s": K_s, "d": K_d, "a": K_a, "": K_h });
```

🐐💬 Create a model for machine learning, the model simply learning networks that respond to our input as we defined and loopback propagation for a simple learning model updates weights as logicals works to replay on target devices. Learning with slope tangent is fast and effective method that is because of working with limited number of object types and fast estimation is importance as accuracy. </br>

```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Initialize
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
input_shape = (5, 1)                                  # 🐑💬 ➰ Create model, optimizer
                                                      # and loss evaluation function and compile

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape),
    
    tf.keras.layers.Reshape((1, 5)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))

])
        
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(192))
model.add(tf.keras.layers.Dense(5, activation = 'softmax'))
model.summary()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Optimizer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=momentum,
    nesterov=False,
    name='SGD',
)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Loss Fn
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""								
lossfn = tf.keras.losses.MeanSquaredError(
    reduction='sum_over_batch_size',
    name='mean_squared_error'
)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Summary
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model.compile(optimizer=optimizer, loss=lossfn, metrics=['accuracy'])
```
🐐💬 File save and reload is important because machine learning sometimes is a long process learning we need to run and replay multiple times, saved logic output also duplicates the process for multiple learning or running machines. </br>
```
history = [];                                          # 🐑💬 ➰ Management file and model training log

checkpoint_path = "/Users/jirayukaewprateep/Applications/TF_Snake_01.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

if not exists(checkpoint_dir) : 
	os.mkdir(checkpoint_dir)
	print("Create directory: " + checkpoint_dir)
 
if exists(checkpoint_path) :
	model.load_weights(checkpoint_path)
	print("model load: " + checkpoint_path)
```

🐑💬 ➰ Random is an important step of learning to shorten processes your random must be simply to understand for machine and the user, Simply distance from one of axes make a strong number times multiple of random dimensions exist arrays will create high costs more chance to be selected. </br>
```
""" : Class / Functions """
def random_action(gamestate):
    temp = tf.random.normal([1,5], 0.2, 0.8, tf.float32)
    # actions = dict({ "w": K_w, "s": K_s, "d": K_d, "a": K_a, "": K_h });

    snake_head_x = 255 - float(extract_valuefrom_gamestate( _gamestate, "snake_head_x" ))
    snake_head_y = 255 - float(extract_valuefrom_gamestate( _gamestate, "snake_head_y" ))
    snake_food_x = 255 - float(extract_valuefrom_gamestate( _gamestate, "food_x" ))
    snake_food_y = 255 - float(extract_valuefrom_gamestate( _gamestate, "food_y" ))

    temp = tf.math.multiply(temp, tf.constant([snake_head_y - snake_food_y, snake_head_y, snake_head_x - snake_food_x, snake_head_x, 0.2], shape=(5,1), dtype=tf.float32));
    temp = tf.nn.softmax(temp[0]);
    action = int(tf.math.argmax(temp))
    action = list(actions.values())[action]
    return action;
```
🐑💬 ➰ Data types conversion is important and use multiple times in the application because of we need to communicate with two platform users and machine learning by a good design they will have a simple matraix return value mapping or data seleted. later use of mapping definition make code more simpler and saved to run because of debugging is possible and more comparible of input and output. </br>
```
def predict_action( DATA ):                           # 🐑💬 ➰ From training model predict result from input value
	
    # temp = DATA[0,:,:,:,:]
    # print( temp.shape )

    predictions = model.predict(DATA)
    score = tf.nn.softmax(predictions[0])
    action = int(tf.math.argmax(score))
    action = list(actions.values())[action]

    return action;

## {'snake_head_x': np.float64(108.0), 'snake_head_y': 192.0, 'food_x': np.int64(38), 'food_y': np.int64(57), 'snake_body': [0.0, 11.0, 22.0], 'snake_body_pos': [[np.float64(108.0), 192.0], [np.float64(97.0), 192.0], [np.float64(86.0), 192.0]]}
def extract_valuefrom_gamestate( gamestate, str_attr ):

    return gamestate[str_attr];

def create_dataset( _gamestate, scores, action ):
    snake_head_x = extract_valuefrom_gamestate( _gamestate, "snake_head_x" )
    snake_head_y = extract_valuefrom_gamestate( _gamestate, "snake_head_y" )
    snake_food_x = extract_valuefrom_gamestate( _gamestate, "food_x" )
    snake_food_y = extract_valuefrom_gamestate( _gamestate, "food_y" )
    
    DATA = tf.constant([scores, snake_head_x, snake_head_y, snake_food_x, snake_food_y ], shape=(1,1,5,1))
    LABEL = tf.constant([action], shape=(1,1,1))  

    

    return DATA, LABEL;

def action_tonumber( action ):
    # actions = dict({ "w": K_w, "s": K_s, "d": K_d, "a": K_a, "": K_h });
    idx_map = {key: i for i, key in enumerate(actions.values())}
    
    return idx_map.get(action)
```
👧💬 🎈 Execution I loved this part that is because it connected of definition and logics we to see the overall process with allows users to response. </br>
```
p.init()
scores = 0.0;
reward = 0.0;
step = 0;
nb_frames = 1000000000
action = 0;

for i in range(nb_frames):
    if p.game_over():
        DATA, LABEL = create_dataset( _gamestate, scores, action_tonumber(action) );
        dataset = tf.data.Dataset.from_tensor_slices((DATA, LABEL)) 
        history = model.fit(dataset, epochs=10)
        model.save_weights(checkpoint_path)

        p.reset_game() 
        scores = 0.0;
        reward = 0.0;
        step = 0;

    _gamestate = p.getGameState();
    print(extract_valuefrom_gamestate( _gamestate, "snake_head_x" ))
    
    if reward < 10 :
        action = random_action(_gamestate);
    else: 
        DATA, LABEL = create_dataset( _gamestate, scores, action_tonumber(K_h) );
        action = predict_action(DATA);

    reward = p.act(action);
    scores = scores + reward;

    DATA, LABEL = create_dataset( _gamestate, scores, action_tonumber(action) );
    dataset = tf.data.Dataset.from_tensor_slices((DATA, LABEL)) 
    history = model.fit(dataset, epochs=10)
```
