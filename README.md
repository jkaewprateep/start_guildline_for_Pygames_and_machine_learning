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
🐯💬 Culture-INFO, The input needs to be displayed or accepted from the platform or Windows user understands while running on VSCode, it displays on VSCode instead, that is how they design to tell the user to run application games outside VSCode. They can run the games with signal and keyinput but it is more likely for debugging. </br>
🦁💬 Python environment had support for many games you can download and play with Gym Pygames. I had a project called Galaxy, it is a shooting game and uses input as virtual game observation ( screen ) as input for the machine learning. </br>

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

🐐💬 Create custom variables for the program and machine learning. The wide usage of variables needs to be updated to include global variables and rare internal variables because the program examines the variable scopes, and for security, the application accesses the same global variables. </br>
🦭💬 In evaluation, they may require an interface file for communication, or they can read from the  communication of the program. Some applications read from the memory of the program by a block of memory defined and update the value from the program application, creating a change in application behavior sometimes allows us to read a value selected or change the dropdown value list. By performing this action should consult with the application owner, and we do this for find bug or debugging program. There are more sample with CTI application integration where they can see communication or allowed list or linked list value they are working in the memory </br>

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
🧸💬 Size of the network does not mean you are smarter or dumber but matching of response and objective, same as Whales and humans. </br>

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
🧸💬 A Good random is not to clear the games with defined matrices but explore action for the machine learning as a player. </br>
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
---
<p align="center" width="100%">
    <img width="30%" src="https://github.com/jkaewprateep/advanced_mysql_topics_notes/blob/main/custom_dataset.png">
    <img width="30%" src="https://github.com/jkaewprateep/advanced_mysql_topics_notes/blob/main/custom_dataset_2.png"> </br>
    <b> 🥺💬 รับจ้างเขียน functions </b> </br>
</p>
