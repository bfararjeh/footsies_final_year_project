# FOOTSIES: Neural Network Edition!

For my final year project, I've decided to implement an advanced lifelike AI utilising neural networks and machine learning to improve upon the existing AI in the game and give players a better singleplayer experience.
This repository is a fork of the main project, and will contain any changes I've made to the original source code, as well as any files created in the process of devloping this AI.
This project can be downloaded and played at your liesure, provided it may be in an incomplete state at any given point in time.
This AI will only work on the current older build of Footsies, as the most recent build is available only for purchase on steam, and does not come with the source code, which was the defining reason for choosing to work on this older branch.

The neural network takes control of player 1, and you can either let it battle the CPU or play against it as player 2.

## Build Instructions

You must have Python installed to run this project. You can install python **here:** https://www.python.org/downloads/
This project is also only available on windows.

With Python installed, follow the instructions below to setup the project:

    - Open the "FOOTSIES-NEURAL-NETWORK-EDITION-V1.0.0\NeuralNetwork" directory in windows file explorer.
    - In the file explorer, click on the directory tree above all the files.
    - Type "cmd", windows command prompt should open, with the current directory set to the neural network folder.
    - In the cmd, type the following:
    
        "python -m pip install -r requirements.txt"

      This will use pip to install all required dependencies.
    - Once pip has finished installing, you're done!

## Operation Instructions

Once the project has been setup, simply executing the "FOOTSIES.exe" file will run the game.
Upon running the game, a Python server will open up. Please wait for confirmation from the server that the network has loaded and the client-server link established (this may take a while on your first launch).
Once the server is up, the game is good to go!

If you would like to take control of player 1 at any given time, simply close the Python server, and control will be restored.

## Contact

My **github:** https://github.com/bfararjeh
My **email:** bahafararjeh@hotmail.com
My **youtube:** https://youtube.com/@fararjeh

Thank you for installing!

Below is the original ReadMe for this project.

# Footsies

FOOTSIES is a 2D fighting game where players can control character movement horizontally 
and use one attack button to perform normal and special moves to defeat their opponent.
While the controls (and graphics) are super simple, 
FOOTSIES retains the fundamental feeling of fighting game genre 
where spacing, hit confirm and whiff punish are keys to achieve victory.

<img class="row-picture" src="https://hifight.github.io/static/img/footsies/footsies_00.jpg">

This is a fun little project for the fighting game community and 
for own practice developing a game by myself.
I am objectively bad at art and music but that went kinda well with theme of the game. 
The animation in this game are, obviously, heavily inspired by the most iconic you-know-who fighting game character.

Although I only tested this game mostly with CPU, when I actually tried this game with my friend, 
we actually had a lot of fun! So, I hope that everyone try this game and have some fun as well :D

<img class="row-picture" src="https://hifight.github.io/static/img/footsies/footsies_01.jpg">

<h3>Download</h3> 

<b><u><a href="https://github.com/hifight/Footsies/releases" download>FOOTSIES</a></u></b>

※You can config keys/buttons input when the game is launched, although XInput can't be set on config windows, 
XInput controller should work fine in the game.


<img class="row-picture" src="https://hifight.github.io/static/img/footsies/footsies_03.jpg">


<h3>Mechanics</h3> 
- There is no health bar. The round is lost after being hit by special moves.
- There is, however, guard bar. You can block opponent attack up to three times. After that, every attack will cause guard break.
- There are two type of normal moves, neutral attack and forward/backward attack.
- There are two type of special moves which can be performed by holding and then release attack button.
One can be performed by neutral release, and forward/backward release for the other one.
- If normal moves connect with the opponent, whether on hit or block, it can be canceled into neutral special move by pressing an attack button again.
- Forward and backward dashes can be performed by pressing forward/backward twice.
- Hitbox/hurtbox/frame information can be toggle on and off by pressing F12.
- Press F1 to pause/resume the game. While pausing, pressing F2 will play the game for 1 frame.


<img class="row-picture" src="https://hifight.github.io/static/img/footsies/footsies_04.jpg">


Whether you like the game or not, feel free to leave some comments about your experience on my <b><u><a href="https://twitter.com/">twitter</a></u></b>

If you like the game then invite your friend to play this game too! Seeing some tournament for this game would be a dream come true for me.
